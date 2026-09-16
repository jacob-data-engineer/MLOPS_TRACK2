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
    // A 401 is a session that ended, not a failure of the call. Clear it and
    // let the app fall back to sign-in; the error is still thrown so an
    // in-flight screen can stop what it was doing.
    //
    // NOT for the login request itself: a wrong password is a 401 the login
    // form must show, and logging out mid-login would wipe the error.
    if (res.status === 401 && path !== "/auth/login" && onUnauthorized) {
      onUnauthorized();
    }
    throw new ApiError(res.status, detail);
  }

  if (res.status === 204) {
    return undefined as T;
  }
  return (await res.json()) as T;
}

export function apiGet<T>(path: string): Promise<T> {
  return request<T>(path, { method: "GET" });
}

export function apiPost<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
}

export function apiPut<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: "PUT",
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
// Model deployment status (/models) -- the "Model deployment
// status" tab, single ModelType.PPE card per the session's consolidation.
// ---------------------------------------------------------------------------

export interface ModelVersion {
  version: string | number;
  deployment_status?: string | null;
  approval_status?: string | null;
  creation_time?: string | null;
  eval_metrics_s3_uri?: string | null;
  [key: string]: unknown;
}

export interface ModelStatusResponse {
  model_type: string;
  latest: ModelVersion | null;
  versions: ModelVersion[];
}

export function getModelStatus(): Promise<ModelStatusResponse> {
  return apiGet<ModelStatusResponse>("/models/status");
}

// ---------------------------------------------------------------------------
// Shared eval-dataset listing shape -- `GET /eval/datasets` and
// `GET /track1/eval-datasets` both return this (see dataset_registry.py):
// named gold sets (v1, then v2) first, every other real case-list file
// after. `DatasetPicker` is the one place this renders.
// ---------------------------------------------------------------------------

export interface EvalDataset {
  key: string;
  label: string;
  is_named: boolean;
}

// ---------------------------------------------------------------------------
// Rolling metrics (/metrics) -- the "Rolling metrics" screen.
// ---------------------------------------------------------------------------

export interface OutcomeCounts {
  CONFIRM_TP: number;
  CONFIRM_FP: number;
  ESCALATE: number;
  [key: string]: number;
}

export interface PerHazardStats {
  total: number;
  outcome_counts: OutcomeCounts;
}

export interface Track1Metrics {
  total_alarms: number;
  outcome_counts: OutcomeCounts;
  auto_annotation_rate: number;
  per_hazard_type: Record<string, PerHazardStats>;
  label_volume_this_week: number;
  /** The same week split by what the labels are. `images` is distinct frames,
   * `person` is boxes holding a person (a row settled as "nobody here" has
   * none), `helmet` is boxes on people the judge cleared. */
  label_volume_breakdown: { images: number; person: number; helmet: number };
  flag_miss_count: number;
  flag_miss_note: string;
}

export type ShadowModeAgreement =
  | { insufficient_data: true; matched_count: number }
  | {
      insufficient_data: false;
      matched_count: number;
      agreement_count: number;
      agreement_rate: number;
    };

export interface RollingMetricsResponse {
  track1_metrics: Track1Metrics;
  /** The 85% benchmark, measured against AUDITS. Of the cases the judge
   * committed to a verdict on, how often it matched the human -- the same
   * definition the gold-set version used, with only the source of the human
   * answer changed, which is what keeps the number comparable to what this
   * tile reported before.
   *
   * Sourced from audits because the Run pipeline path exists to rehearse the
   * live feed, and a live alarm has no gold label: scoring it against a gold
   * set would measure something the real thing can never reproduce. Gold
   * stays where a gold label genuinely exists -- scoring an eval run against
   * ground truth, on the Eval screen. */
  shadow_mode_agreement: ShadowModeAgreement;
  /** Always `"audit"` today; present so the UI states what it is measuring
   * rather than asserting a source it cannot see. */
  shadow_mode_agreement_source: string;
  /** The same judge measured the old way, against reviewers' gold labels.
   * Reported for reference, not as the benchmark: where both exist the GAP is
   * the interesting part, since an auditor sees the pipeline's answer before
   * agreeing with it and a gold labeller did not. */
  gold_set_agreement: ShadowModeAgreement;
  shadow_mode_agreement_target: number;
  label_volume_weekly_target: number;
  track2_retraining_done: boolean;
  /** The audit decisions the benchmark above is computed from -- agreement by
   * origin, plus `confusion` (precision/recall/F1/accuracy). Declared with the
   * rest of the audit types further down this file, beside the queue they are
   * recorded from.
   *
   * Not a second opinion on the same number: `shadow_mode_agreement` is this
   * block's `confusion.overall` accuracy read out loud. If the two ever
   * disagreed, the page would be making two claims about one thing. */
  audit: AuditSummary;
}

export function getRollingMetrics(): Promise<RollingMetricsResponse> {
  return apiGet<RollingMetricsResponse>("/metrics/rolling");
}

export function listEvalRuns(): Promise<{ runs: string[] }> {
  return apiGet<{ runs: string[] }>("/metrics/eval-runs");
}

// How the judge answered. THE KEYS KEEP THEIR HISTORICAL NAMES: the server
// has always keyed these `hold` / `cleared` / `escalate`, and the UI now
// stamps those same three answers VIOLATION / NO VIOLATION / ESCALATE (see
// VerdictStamp's `VERDICT_WORD`, the one place that mapping lives). Renaming
// the keys would be a wire-format change for a wording decision, so it was
// deliberately not done -- the mismatch below is explained, not accidental.
// Always sums to the matching ground_truth_*_n.
export interface VerdictDistribution {
  hold: number;
  cleared: number;
  escalate: number;
}

export interface EvalReportRow {
  label: string;
  violation_type: string;
  n: number;
  precision: number;
  recall: number;
  f1: number;
  accuracy: number;
  uncertain_rate: number;
  // The rate columns say how often the judge was right; these say which way
  // it went when it was wrong. A recall of 0.8 hides whether the missing 20%
  // was stamped NO VIOLATION outright or escalated to a human -- very
  // different failures on a real violation.
  ground_truth_violation_n: number;
  violation_distribution: VerdictDistribution;
  ground_truth_no_violation_n: number;
  no_violation_distribution: VerdictDistribution;
  raw_cv_precision: number;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  reasoning_alignment_mean: number | null;
  reasoning_scored_count: number;
  reasoning_total_count: number;
}

// One scored group per (run file, method). A multi-method run streams every
// method into a single jsonl tagged with the method label, so the server
// splits on that tag rather than folding the file into one report -- an
// average across methods that were run precisely to be compared describes
// none of them. Separate files stay separate for the same reason.
export interface EvalRunGroup {
  filename: string;
  method: string;
  rows: EvalReportRow[];
}

/** One case behind a distribution cell, straight from the run file. */
export interface EvalRunCase {
  case_id: string;
  violation_type: string;
  verdict: string;
  confidence: string | null;
  explanation: string | null;
  /** null for runs recorded before the score was written to the jsonl --
   * "never captured", not "scored nothing". */
  reasoning_score: number | null;
}

/** The distribution keys above, as a type. Still `hold` / `cleared` /
 * `escalate` on the wire; the words a reader sees come from
 * `VerdictStamp`. */
export type VerdictStampKey = "hold" | "cleared" | "escalate";

/** The cases behind one cell of a scored report: `ground_truth` picks what
 * the cases really were, `stamp` picks what the model said about them. So
 * ("TP", "cleared") returns the real violations the model stamped NO
 * VIOLATION, which are the missed hazards. The `stamp` values are the
 * server's historical keys, not the words on screen -- see
 * `VerdictDistribution`. */
export function getEvalRunCases(
  filename: string,
  method: string,
  groundTruth: "TP" | "FP",
  stamp: VerdictStampKey,
): Promise<{ cases: EvalRunCase[] }> {
  const q = new URLSearchParams({
    filename,
    method,
    ground_truth: groundTruth,
    stamp,
  });
  return apiGet<{ cases: EvalRunCase[] }>(`/metrics/eval-runs/cases?${q}`);
}

/** Path for `useApiImage`. `crop` should match the inspected method's
 * use_crop, so the picture is what the model actually saw. */
export function evalRunCaseImagePath(caseId: string, crop: boolean): string {
  const q = new URLSearchParams({ case_id: caseId, crop: String(crop) });
  return `/metrics/eval-runs/case-image?${q}`;
}

export function combineEvalRuns(
  filenames: string[],
): Promise<{ groups: EvalRunGroup[] }> {
  return apiPost<{ groups: EvalRunGroup[] }>("/metrics/eval-runs/combine", {
    filenames,
  });
}

// ---------------------------------------------------------------------------
// Few-shot library (/fewshot) -- the "Few-shot library review" screen.
// The 5 ViolationType values (ppe_pipeline/models.py) are fixed, same as
// Layout.tsx's hardcoded 6 nav domains -- there's no dedicated
// "list violation types" endpoint under /fewshot to source this from.
// ---------------------------------------------------------------------------

export const FEWSHOT_VIOLATION_TYPES = [
  "glove",
  "helmet",
  "dangerous_plane",
  "too_near_forklift",
  "suspended_load",
] as const;
export type FewshotViolationType = (typeof FEWSHOT_VIOLATION_TYPES)[number];

export interface FewshotLibraryChoice {
  name: "fixed" | "sampled";
  label: string;
  manifest_path: string;
  annotations_path: string;
  /** A sampled library is derived: rebuilt whole at the start of every
   * sampled run, so the server refuses edits to it. */
  editable: boolean;
  available: boolean;
}

export interface FewshotDefaultPaths {
  manifest_path: string;
  annotations_path: string;
  /** The libraries a run can be given. A run picks one, so this screen has
   * to be able to show the one a run actually used. */
  libraries: FewshotLibraryChoice[];
}

export function getFewshotDefaultPaths(): Promise<FewshotDefaultPaths> {
  return apiGet<FewshotDefaultPaths>("/fewshot/paths/defaults");
}

export interface FewshotLoadResponse {
  manifest_path: string;
  annotations_path: string;
  counts: Record<string, number>;
}

/** `GET /fewshot/status` -- never errors, so this is safe to call
 * unconditionally on mount to check "is a library already loaded this
 * session?" without a 404 round-trip first (the library itself is durable
 * server-side across a navigate-away-and-back, unlike this page's own
 * local `loaded` state before this build). */
export type FewshotStatus =
  | { loaded: false; manifest_path: null; annotations_path: null; counts: null }
  | { loaded: true; manifest_path: string; annotations_path: string; counts: Record<string, number> };

export function getFewshotStatus(): Promise<FewshotStatus> {
  return apiGet<FewshotStatus>("/fewshot/status");
}

export function loadFewshotLibrary(
  manifestPath: string,
  annotationsPath: string,
): Promise<FewshotLoadResponse> {
  return apiPost<FewshotLoadResponse>("/fewshot/load", {
    manifest_path: manifestPath,
    annotations_path: annotationsPath,
  });
}

export interface FewshotEntry {
  index: number;
  image: string;
  verdict: string;
  reason: string;
  confidence: string;
  explanation: string;
}

export interface FewshotEntriesResponse {
  entries: FewshotEntry[];
  tp_count: number;
  fp_count: number;
  uncertain_count: number;
  reason_options: string[];
}

export function listFewshotEntries(violationType: string): Promise<FewshotEntriesResponse> {
  return apiGet<FewshotEntriesResponse>(
    `/fewshot/entries?violation_type=${encodeURIComponent(violationType)}`,
  );
}

export function fewshotEntryImagePath(violationType: string, index: number): string {
  return `/fewshot/entries/${encodeURIComponent(violationType)}/${index}/image`;
}

export interface FewshotEntryFormValue {
  verdict: string;
  reason: string;
  confidence: string;
  explanation: string;
}

export function updateFewshotEntry(
  violationType: string,
  index: number,
  payload: FewshotEntryFormValue,
): Promise<FewshotEntry> {
  return apiPut<FewshotEntry>(
    `/fewshot/entries/${encodeURIComponent(violationType)}/${index}`,
    payload,
  );
}

export function removeFewshotEntry(
  violationType: string,
  index: number,
): Promise<{ removed: string }> {
  return apiDelete<{ removed: string }>(
    `/fewshot/entries/${encodeURIComponent(violationType)}/${index}`,
  );
}

export function listAvailableFewshotCrops(violationType: string): Promise<{ available: string[] }> {
  return apiGet<{ available: string[] }>(
    `/fewshot/crops/available?violation_type=${encodeURIComponent(violationType)}`,
  );
}

export function addFewshotFromExistingCrop(
  violationType: string,
  existingFilename: string,
  form: FewshotEntryFormValue,
): Promise<{ added: string }> {
  return apiPost<{ added: string }>("/fewshot/entries/from-existing-crop", {
    violation_type: violationType,
    existing_filename: existingFilename,
    ...form,
  });
}

export function addFewshotFromUpload(
  violationType: string,
  image: File,
  form: FewshotEntryFormValue,
): Promise<{ added: string }> {
  const params = new URLSearchParams({
    violation_type: violationType,
    verdict: form.verdict,
    reason: form.reason,
    confidence: form.confidence,
    explanation: form.explanation,
  });
  const formData = new FormData();
  formData.append("image", image);
  return apiPostForm<{ added: string }>(`/fewshot/entries/upload?${params.toString()}`, formData);
}

// ---------------------------------------------------------------------------
// Track1 alarm review (/track1) -- the "Run pipeline" screen.
// Scoped in this pass to the dataset-based alarm source only (no live
// camera feed exists anywhere in this codebase to build an upload/manual
// flow against yet -- see the architect's earlier finding).
// ---------------------------------------------------------------------------

/** `GET /track1/hazard-types` is deprecated -- it now just mirrors
 * `GET /eval/violation-types`. Kept here only if some caller still needs
 * the old shape; `Track1Page` sources its violation-type selector from
 * `listEvalViolationTypes()` instead. */
export function getTrack1HazardTypes(): Promise<{ hazard_types: string[] }> {
  return apiGet<{ hazard_types: string[] }>("/track1/hazard-types");
}

export interface Track1ConfigResponse {
  provider: string;
  model: string;
  model_options: Record<string, string[]>;
}

export function getTrack1Config(): Promise<Track1ConfigResponse> {
  return apiGet<Track1ConfigResponse>("/track1/config");
}

export function listTrack1EvalDatasets(): Promise<{ datasets: EvalDataset[] }> {
  return apiGet<{ datasets: EvalDataset[] }>("/track1/eval-datasets");
}

export interface Track1CaseSummary {
  case_id: string;
  violation_type: string;
}

/** `counts` is per violation type over the whole file, so the batch control
 * can say how many cases a selection covers BEFORE spending a judge call on
 * any of them. */
export interface Track1CasesResponse {
  cases: Track1CaseSummary[];
  counts: Record<string, number>;
}

export function listTrack1EvalDatasetCases(path: string): Promise<Track1CasesResponse> {
  return apiGet<Track1CasesResponse>(
    `/track1/eval-datasets/cases?path=${encodeURIComponent(path)}`,
  );
}

export interface Track1Detection {
  violation_type: string;
  person_box: [number, number, number, number];
  source: string;
  confidence: number;
  region_box: [number, number, number, number] | null;
  forklift_box: [number, number, number, number] | null;
}

export interface Track1CaseResponse {
  case_id: string;
  media_type: string;
  detections: Track1Detection[];
}

export function getTrack1EvalDatasetCase(
  path: string,
  caseId: string,
): Promise<Track1CaseResponse> {
  return apiGet<Track1CaseResponse>(
    `/track1/eval-datasets/case?path=${encodeURIComponent(path)}&case_id=${encodeURIComponent(caseId)}`,
  );
}

export function track1EvalDatasetCaseImagePath(path: string, caseId: string): string {
  return `/track1/eval-datasets/case-image?path=${encodeURIComponent(path)}&case_id=${encodeURIComponent(caseId)}`;
}

export interface RunTrack1Params {
  image: Blob;
  filename: string;
  mediaType: string;
  violationType: string;
  personBox: [number, number, number, number];
  forkliftBox?: [number, number, number, number] | null;
  regionBox?: [number, number, number, number] | null;
  alarmId?: string;
  sourceDetectionId?: string;
  overrideProvider?: string | null;
  overrideModel?: string | null;
  /** What draws the helmet box when the judge finds no violation on a
   * helmet alarm:
   * `"cv"` (the default) runs the stage-2 helmet detector; `"vlm"` skips it
   * so the judge's own inline box stands and `helmet_box_source` comes back
   * `"vlm"`. The VLM path is the automatic fallback anyway -- this only
   * forces it, which is how a tester deliberately produces the loose boxes
   * the review queue exists to catch (the detector clears most cases well
   * above the review threshold, so they're otherwise hard to come by).
   * Omitted means the server's own `"cv"` default. */
  helmetBoxMode?: HelmetBoxMode | null;
}

/** The two things that can draw a helmet box, as `POST /track1/run` and
 * `/track1/run-job` accept it in `helmet_box_mode`. Same vocabulary as
 * `HelmetBoxSource`, which is what comes back out. */
export type HelmetBoxMode = "cv" | "vlm";

export interface RunTrack1Response {
  alarm_id: string;
  hazard_type: string;
  violation_type: string;
  outcome: string;
  verdict: string;
  confidence_level: string;
  confidence_score: number | null;
  reason: string;
  explanation: string;
  /** The crop the judge was actually shown, base64 PNG, with the green
   * helmet box already drawn into it when the judge found no violation on
   * one. Inlined rather than fetched: this screen judges a single image, so
   * one small PNG in the response beats a second endpoint. */
  judged_image_png: string | null;
  /** `[x1, y1, x2, y2]` -- in the judged crop's own pixels, and on the
   * original frame. Both null unless the judge answered a helmet alarm with
   * `helmet_worn`, its no-violation reason. */
  helmet_box_crop: number[] | null;
  helmet_box_original: number[] | null;
  /** The stage-2 helmet detector's confidence in this box. Below
   * `HELMET_BOX_REVIEW_CONFIDENCE` the box is worth looking at but not worth
   * training on -- it never affects the verdict. Null whenever
   * `helmet_box_source` is not "cv", since only the detector reports a score. */
  helmet_box_confidence: number | null;
  /** Which model drew this box: "cv" (the helmet detector) or "vlm" (the
   * judge's own inline box, the fallback). Null when there is no box. */
  helmet_box_source: HelmetBoxSource;
}

function buildTrack1FormData(params: RunTrack1Params): FormData {
  const formData = new FormData();
  formData.append("image", params.image, params.filename);
  formData.append("media_type", params.mediaType);
  formData.append("violation_type", params.violationType);
  formData.append("person_box", JSON.stringify(params.personBox));
  if (params.forkliftBox) formData.append("forklift_box", JSON.stringify(params.forkliftBox));
  if (params.regionBox) formData.append("region_box", JSON.stringify(params.regionBox));
  if (params.alarmId) formData.append("alarm_id", params.alarmId);
  if (params.sourceDetectionId) formData.append("source_detection_id", params.sourceDetectionId);
  if (params.overrideProvider) formData.append("override_provider", params.overrideProvider);
  if (params.overrideModel) formData.append("override_model", params.overrideModel);
  if (params.helmetBoxMode) formData.append("helmet_box_mode", params.helmetBoxMode);
  return formData;
}

/** Superseded by `startTrack1RunJob` below (kept in place, unused by
 * `Track1Page` after this build -- see CONTRACT.md). */
export function runTrack1Inference(params: RunTrack1Params): Promise<RunTrack1Response> {
  return apiPostForm<RunTrack1Response>("/track1/run", buildTrack1FormData(params));
}

// ---------------------------------------------------------------------------
// Track1 "Run pipeline" background job -- `/track1/run-job`,
// `/track1/run-job/progress`. Its own session-key slot, independent of
// every eval job below -- see CONTRACT.md's "Track1 'Run pipeline'
// background job" section. A single judge call only takes seconds; this
// exists purely so "Run pipeline" follows the same job-poll shape as every
// other long-running action in the app, per the CEO's consistency call.
// ---------------------------------------------------------------------------

export function startTrack1RunJob(params: RunTrack1Params): Promise<{ job_id: string; total: number }> {
  return apiPostForm<{ job_id: string; total: number }>("/track1/run-job", buildTrack1FormData(params));
}

/** `GET /track1/run-job/progress` -- discriminated on `status`. `"none"`
 * means no job has ever been started this session (safe to call
 * unconditionally on mount, to resume a running/finished job after a
 * reload/navigate-away-and-back). */
export type Track1JobProgress =
  | { status: "none" }
  | {
      status: "running" | "done" | "error";
      job_id: string;
      total: number;
      processed: 0 | 1;
      result: RunTrack1Response | null;
      error: string | null;
    };

export function getTrack1RunJobProgress(): Promise<Track1JobProgress> {
  return apiGet<Track1JobProgress>("/track1/run-job/progress");
}

// ---------------------------------------------------------------------------
// Track1 BATCH run -- `/track1/batch-run`, `/track1/batch-run/progress`,
// `/track1/batch-run/cancel`. Runs every case of the chosen violation types
// through the real judge, mirroring the eval screen's multi-select but
// writing to the audit log the escalation and audit queues read. Its own
// session-key slot again, independent of the single-case job above.
// ---------------------------------------------------------------------------

export interface StartTrack1BatchRunParams {
  dataset_path: string;
  violation_types: string[];
  helmet_box_mode?: HelmetBoxMode;
  /** Cap on cases per run. The server defaults to 50; a judge call costs
   * money and takes seconds, so an unbounded "all violations" on a
   * 1848-case dataset is not something a mis-click should be able to start. */
  limit?: number;
}

export function startTrack1BatchRun(
  params: StartTrack1BatchRunParams,
): Promise<{ job_id: string; total: number }> {
  return apiPost<{ job_id: string; total: number }>("/track1/batch-run", params);
}

export interface Track1BatchRow {
  case_id: string;
  alarm_id: string;
  violation_type: string | null;
  outcome: string;
  /** The judge's confidence in the VERDICT. Not the same question as
   * `helmet_box_confidence`, which is the detector's confidence in the box:
   * a case can carry a certain verdict and a doubtful box, or the reverse. */
  confidence_score: number | null;
  confidence_level: string | null;
  reason: string | null;
  helmet_box_source: string | null;
  helmet_box_confidence: number | null;
}

/** Discriminated on `status`, like the single-case job. `processed` counts
 * ATTEMPTS, so it reaches `total` whether or not every case produced a row --
 * `rows.length + errors.length` is what accounts for all of them. */
export type Track1BatchProgress =
  | { status: "none" }
  | {
      status: "running" | "done" | "cancelled" | "error";
      job_id: string;
      total: number;
      processed: number;
      helmet_box_mode: HelmetBoxMode | null;
      rows: Track1BatchRow[];
      errors: { case_id: string; error: string }[];
    };

export function getTrack1BatchProgress(): Promise<Track1BatchProgress> {
  return apiGet<Track1BatchProgress>("/track1/batch-run/progress");
}

/** One already-judged case, in full -- read from the audit log rather than
 * from a session job, so it works after a reload and for any case ever
 * judged, not only this session's batch. Every box is in ORIGINAL FRAME
 * PIXELS alongside the `crop_box` they were measured in: a caller drawing
 * over the cropped image subtracts the crop origin itself. */
export interface JudgedCase {
  alarm_id: string;
  hazard_type: string | null;
  timestamp: string | null;
  outcome: string;
  reason: string | null;
  /** The judge's own words. `reasoning` and `explanation` are the same
   * string -- the audit log calls it one thing, the run response the other. */
  reasoning: string | null;
  explanation: string | null;
  confidence_score: number | null;
  confidence_level: string | null;
  person_box: [number, number, number, number] | null;
  crop_box: [number, number, number, number] | null;
  helmet_box: [number, number, number, number] | null;
  helmet_box_source: HelmetBoxSource;
  helmet_box_confidence: number | null;
  image_width: number | null;
  image_height: number | null;
  judge_model: string | null;
  judge_provider: string | null;
  helmet_detector_weights: string | null;
  /** A legacy row can name a frame whose bytes were never stored, so
   * "has a key" is not "has a frame". */
  frame_available: boolean;
  /** False means `/image` serves the WHOLE frame -- better than nothing, but
   * not what a reviewer can place a box on. */
  crop_available: boolean;
}

export function getJudgedCase(alarmId: string): Promise<JudgedCase> {
  return apiGet<JudgedCase>(`/track1/judged/${encodeURIComponent(alarmId)}`);
}

/** Path for `useApiImage` -- the judged frame, unannotated, cropped to what
 * the judge actually saw. */
export function judgedCaseImagePath(alarmId: string, crop = true): string {
  return `/track1/judged/${encodeURIComponent(alarmId)}/image?crop=${crop}`;
}

/** Cooperative: the case in flight finishes and its result is kept, because
 * that judge call has already been paid for. */
export function cancelTrack1BatchRun(): Promise<{
  status: string;
  processed: number;
  total: number;
}> {
  return apiPost<{ status: string; processed: number; total: number }>(
    "/track1/batch-run/cancel",
    {},
  );
}

export interface Track1RollingResultRow {
  alarm_id: string;
  hazard_type: string;
  violation_type: string;
  outcome: string;
  confidence_level: string;
  confidence_score: number | null;
  reason: string;
}

export interface Track1SessionResponse {
  rolling_results: Track1RollingResultRow[];
  calls: number;
}

export function getTrack1Session(): Promise<Track1SessionResponse> {
  return apiGet<Track1SessionResponse>("/track1/session");
}

export function clearTrack1Session(): Promise<Track1SessionResponse> {
  return apiPost<Track1SessionResponse>("/track1/session/clear");
}

// ---------------------------------------------------------------------------
// Bulk eval + one-by-one stepper (/eval) -- the "Evaluate
// against ground truth" tab. There's no dedicated /eval config endpoint for
// a default provider/model (unlike Track1's /track1/config) -- this page
// prefills from getTrack1Config() too, since it's the same JudgeClient/
// config machinery app.py's one sidebar selector feeds every tab.
// ---------------------------------------------------------------------------

export interface EvalMethodConfig {
  shot: "zero" | "few";
  use_crop: boolean;
  bypass_forklift_filter: boolean;
  label: string;
}

export function listEvalDatasets(): Promise<{ datasets: EvalDataset[] }> {
  return apiGet<{ datasets: EvalDataset[] }>("/eval/datasets");
}

export function listEvalMethods(): Promise<{ methods: Record<string, EvalMethodConfig> }> {
  return apiGet<{ methods: Record<string, EvalMethodConfig> }>("/eval/methods");
}

export function listEvalViolationTypes(): Promise<{ violation_types: string[] }> {
  return apiGet<{ violation_types: string[] }>("/eval/violation-types");
}

export function getEvalRuleText(): Promise<{ rule_text: Record<string, string> }> {
  return apiGet<{ rule_text: Record<string, string> }>("/eval/rule-text");
}

export function listEvalPastRuns(): Promise<{ runs: string[] }> {
  return apiGet<{ runs: string[] }>("/eval/runs");
}

export function getEvalPastRun(filename: string): Promise<{ rows: Record<string, unknown>[] }> {
  return apiGet<{ rows: Record<string, unknown>[] }>(`/eval/runs/${encodeURIComponent(filename)}`);
}

export interface RunEvalRequest {
  dataset_path: string;
  methods: string[];
  also_run_glove_crop: boolean;
  violation_types?: string[] | null;
  auto_balance: boolean;
  balance_seed: number;
  few_shot_manifest_path?: string | null;
  provider: string;
  model: string;
  config_path?: string | null;
}

export interface BulkReviewRow {
  case_id: string;
  label: string;
  violation_type: string;
  ground_truth: "TP" | "FP";
  verdict: string;
  confidence: string;
  explanation: string;
  ground_truth_rationale: string | null;
  reasoning_score: number | null;
  /** Where the judge said the helmet is, on a helmet case it answered
   * `helmet_worn` (no violation) -- `[x1, y1, x2, y2]`. `helmet_box_crop` is in the pixels
   * of the image the judge was shown, which is what the crop on screen
   * displays; `helmet_box_original` is the same box on the original frame,
   * which is what still means something away from that crop. Both null on
   * every other case, including a helmet the judge called a violation. */
  helmet_box_crop: number[] | null;
  helmet_box_original: number[] | null;
  /** The stage-2 helmet detector's confidence in this box. Below
   * `HELMET_BOX_REVIEW_CONFIDENCE` the box is worth looking at but not worth
   * training on -- it never affects the verdict. Null whenever
   * `helmet_box_source` is not "cv", since only the detector reports a score. */
  helmet_box_confidence: number | null;
  /** Which model drew this box: "cv" (the helmet detector) or "vlm" (the
   * judge's own inline box, the fallback). Null when there is no box. */
  helmet_box_source: HelmetBoxSource;
}

export type GloveCropReportRow = EvalReportRow & { variant: string };

export type RunEvalResponse =
  | {
      mode: "bulk_review_pending";
      review_rows: BulkReviewRow[];
      warnings: string[];
      results_path: string;
    }
  | {
      mode: "final";
      reports: Record<string, EvalReportRow[]>;
      combined_rows?: EvalReportRow[];
      glove_crop_rows?: GloveCropReportRow[];
      warnings: string[];
      results_path: string;
    };

/** Superseded by `startAdvancedRun` below (kept in place, unused by
 * `EvalPage` after this build -- see CONTRACT.md). */
export function runEval(payload: RunEvalRequest): Promise<RunEvalResponse> {
  return apiPost<RunEvalResponse>("/eval/run", payload);
}

// ---------------------------------------------------------------------------
// Advanced-mode background job -- `/eval/advanced-run`, runs the exact same
// thing `POST /eval/run` runs, but on a background thread polled via
// `GET /eval/advanced-run/progress`, in its own session-key slot entirely
// independent of Simple mode's `/eval/bulk-run` job -- see CONTRACT.md's
// "Advanced-mode background job" section.
// ---------------------------------------------------------------------------

// `warnings` carries the pre-run notes -- few-shot de-duplication, cases
// excluded for leaking into the few-shot library, and auto-balance capping.
// They are returned up front here (and repeated on the progress poll)
// because the run itself starts in the background: without this the
// exclusions would happen silently.
export function startAdvancedRun(
  payload: RunEvalRequest,
): Promise<{ job_id: string; total: number; warnings: string[] }> {
  return apiPost<{ job_id: string; total: number; warnings: string[] }>("/eval/advanced-run", payload);
}

/** `GET /eval/advanced-run/progress` -- discriminated on `status`. `"none"`
 * means no job has ever been started this session. `final` matches
 * `RunEvalResponse`'s `"final"` branch exactly once populated (multi-method
 * and/or glove-crop runs); for a single-method, no-glove-crop run it stays
 * `null` and `session["eval_bulk_report"]` is populated server-side
 * instead -- callers should fetch that the same way via
 * `getPendingBulkReview()` once `status === "done"` with `final === null`. */
export type AdvancedJobProgress =
  | { status: "none" }
  | {
      status: "running" | "done" | "error" | "cancelled";
      job_id: string;
      current_method: string | null;
      methods_done: string[];
      total: number;
      processed: number;
      rows: BulkReviewRow[];
      final: Extract<RunEvalResponse, { mode: "final" }> | null;
      error: string | null;
      results_path_name: string;
    };

export function getAdvancedRunProgress(): Promise<AdvancedJobProgress> {
  return apiGet<AdvancedJobProgress>("/eval/advanced-run/progress");
}

export function cancelAdvancedRun(): Promise<{ job_id: string; status: string }> {
  return apiPost<{ job_id: string; status: string }>("/eval/advanced-run/cancel");
}

/** No auth-free `<img src>` here either (see `useApiImage`) -- one
 * already-judged case's crop from the currently tracked Advanced-mode job,
 * reading from its own `case_outcomes` slot (separate from Simple mode's,
 * per CONTRACT.md). `404` if the job doesn't exist for this session or
 * hasn't reached this case yet. */
export function advancedRunCaseImagePath(caseId: string): string {
  return `/eval/advanced-run/case-image?case_id=${encodeURIComponent(caseId)}`;
}

export function getPendingBulkReview(): Promise<{ review_rows: BulkReviewRow[] }> {
  return apiGet<{ review_rows: BulkReviewRow[] }>("/eval/bulk-review");
}

export function finalizeBulkReview(
  corrections: Record<string, string>,
): Promise<{ rows: EvalReportRow[]; synced_files: string[] }> {
  return apiPost<{ rows: EvalReportRow[]; synced_files: string[] }>("/eval/bulk-review/finalize", {
    corrections,
  });
}

export function discardBulkReview(): Promise<{ discarded: boolean }> {
  return apiPost<{ discarded: boolean }>("/eval/bulk-review/discard");
}

// ---------------------------------------------------------------------------
// Simple-mode bulk run (observable) -- `POST /eval/bulk-run` starts the
// fixed "Few-shot / filter / cropped" method on a background thread and
// returns immediately; the frontend polls `GET /eval/bulk-review/progress`
// instead of blocking on the request, so a slow 320-case run doesn't strand
// the tab mid-fetch and survives a navigate-away-and-back. Entirely separate
// from `POST /eval/run` (Advanced mode, above) -- see CONTRACT.md's
// "Simple-mode bulk run (observable)" section.
// ---------------------------------------------------------------------------

export interface BulkRunRequest {
  dataset_path: string;
  few_shot_manifest_path: string;
  provider: string;
  model: string;
  config_path?: string | null;
}

export function startBulkRun(
  payload: BulkRunRequest,
): Promise<{ job_id: string; total: number; warnings: string[] }> {
  return apiPost<{ job_id: string; total: number; warnings: string[] }>("/eval/bulk-run", payload);
}

/** `GET /eval/bulk-review/progress` -- discriminated on `status`. `"none"`
 * means nothing has ever run this session (safe to call unconditionally,
 * including on mount, to resume a running/finished job after a reload);
 * the other four statuses share one shape -- `rows` grows one entry per
 * judged case, `final_rows` stays `null` until `status === "done"`. */
export type BulkJobProgress =
  | { status: "none" }
  | {
      status: "running" | "done" | "error" | "cancelled";
      job_id: string;
      label: string;
      total: number;
      processed: number;
      rows: BulkReviewRow[];
      final_rows: EvalReportRow[] | null;
      error: string | null;
      results_path_name: string;
    };

export function getBulkRunProgress(): Promise<BulkJobProgress> {
  return apiGet<BulkJobProgress>("/eval/bulk-review/progress");
}

export function cancelBulkRun(): Promise<{ job_id: string; status: string }> {
  return apiPost<{ job_id: string; status: string }>("/eval/bulk-review/cancel");
}

/** No auth-free `<img src>` here either (see `useApiImage`) -- one
 * already-judged case's crop from the currently tracked bulk-run job, its
 * judged box(es) already drawn server-side. `404` if the job doesn't exist
 * for this session or hasn't reached this case yet. */
export function bulkReviewCaseImagePath(caseId: string): string {
  return `/eval/bulk-review/case-image?case_id=${encodeURIComponent(caseId)}`;
}

export interface StartStepperRequest {
  dataset_path: string;
  few_shot_manifest_path: string;
  violation_types?: string[] | null;
  auto_balance: boolean;
  balance_seed: number;
  provider: string;
  model: string;
  config_path?: string | null;
}

export function startStepper(
  payload: StartStepperRequest,
): Promise<{ total: number; idx: number; warnings: string[] }> {
  return apiPost<{ total: number; idx: number; warnings: string[] }>("/eval/stepper/start", payload);
}

export type StepperStateResponse =
  | { done: true; idx: number; total: number; report_rows: EvalReportRow[] }
  | {
      done: false;
      idx: number;
      total: number;
      confirmed_count: number;
      case_id: string;
      violation_type: string;
      current_ground_truth: "TP" | "FP";
      ground_truth_rationale: string | null;
      judge_verdict: string;
      judge_confidence: string;
      judge_explanation: string;
      reasoning_score: number | null;
      reasoning_critique: string | null;
    };

export function getStepperState(): Promise<StepperStateResponse> {
  return apiGet<StepperStateResponse>("/eval/stepper/state");
}

/** No auth-free `<img src>` here either (see `useApiImage`) -- the current
 * case's judged crop, boxes already drawn server-side. */
/** Confidence floor below which a helmet box is flagged as likely needing
 * human review rather than treated as a retraining-label candidate.
 *
 * Mirrors `helmet_box.review_confidence` in config.toml, which is the source
 * of truth -- the server applies it when deciding what to persist; this copy
 * only drives the wording next to the number. Was previously an unnamed 0.5
 * duplicated across both screens that show a box.
 *
 * This is about the BOX, not the case: a low-confidence box never changes
 * the judge's verdict. */
export const HELMET_BOX_REVIEW_CONFIDENCE = 0.5;

/** Which model produced a helmet box. "cv" is the stage-2 helmet detector;
 * "vlm" is the judge's own inline `helmet_box`, used when the detector is
 * disabled, has no weights, cannot load, or found nothing above threshold.
 * The distinction matters to a reviewer: only "cv" carries a confidence
 * score, and "vlm" boxes are markedly looser. */
export type HelmetBoxSource = "cv" | "vlm" | null;

/** Reviewer-facing label for a helmet box's provenance. */
export function helmetBoxSourceLabel(source: HelmetBoxSource): string | null {
  if (source === "cv") return "CV helmet detector";
  if (source === "vlm") return "VLM fallback: judge's inline box, no score";
  return null;
}

export const EVAL_STEPPER_IMAGE_PATH = "/eval/stepper/image";

export function confirmStepper(
  groundTruth: "TP" | "FP",
): Promise<{ idx: number; total: number; done: boolean; synced_files: string[] }> {
  return apiPost<{ idx: number; total: number; done: boolean; synced_files: string[] }>(
    "/eval/stepper/confirm",
    { ground_truth: groundTruth },
  );
}

export function cancelStepper(): Promise<{ cancelled: boolean }> {
  return apiPost<{ cancelled: boolean }>("/eval/stepper/cancel");
}

// ---------------------------------------------------------------------------
// Gold Set dual review (/goldset) -- mirrors app.py's "Gold Set review" tab.
// Role/rotation/owned-fields are always server-computed (gold_set.role_for)
// and returned per case -- this file never reimplements that logic client-
// side, only renders what the server already decided.
// ---------------------------------------------------------------------------

export interface GoldsetOptions {
  reviewers: string[];
  gold_labels: string[];
  evidence_sufficiency: string[];
  difficulty: string[];
  adjudication_status: string[];
}

export function getGoldsetOptions(): Promise<GoldsetOptions> {
  return apiGet<GoldsetOptions>("/goldset/options");
}

export function mergeGoldsetDatasets(
  basePath: string,
  addPath: string,
  outputPath: string,
): Promise<{ total: number; base_count: number; added_count: number }> {
  return apiPost<{ total: number; base_count: number; added_count: number }>("/goldset/merge", {
    base_path: basePath,
    add_path: addPath,
    output_path: outputPath,
  });
}

export interface GoldsetCompletionRow {
  violation_type: string;
  done: number;
  total: number;
}

export function loadGoldset(
  datasetPath: string,
  annotationsPath: string,
): Promise<{ count: number; completion: GoldsetCompletionRow[] }> {
  return apiPost<{ count: number; completion: GoldsetCompletionRow[] }>("/goldset/load", {
    dataset_path: datasetPath,
    annotations_path: annotationsPath,
  });
}

/** `GET /goldset/status` -- never errors, same "check before rendering the
 * 'load to begin' screen" purpose as `getFewshotStatus` above. */
export type GoldsetStatus =
  | { loaded: false; dataset_path: null; annotations_path: null; count: null; completion: null }
  | {
      loaded: true;
      dataset_path: string;
      annotations_path: string;
      count: number;
      completion: GoldsetCompletionRow[];
    };

export function getGoldsetStatus(): Promise<GoldsetStatus> {
  return apiGet<GoldsetStatus>("/goldset/status");
}

export interface ListGoldsetCasesParams {
  violationType?: string;
  adjudicationStatus?: "all" | "disagreements" | "pending";
  reviewer?: string;
  queueMode?: "unfinished" | "all";
}

export function listGoldsetCases(
  params: ListGoldsetCasesParams,
): Promise<{ case_ids: string[]; count: number; completion: GoldsetCompletionRow[] }> {
  const q = new URLSearchParams({
    violation_type: params.violationType ?? "All",
    adjudication_status: params.adjudicationStatus ?? "all",
    reviewer: params.reviewer ?? "",
    queue_mode: params.queueMode ?? "unfinished",
  });
  return apiGet<{ case_ids: string[]; count: number; completion: GoldsetCompletionRow[] }>(
    `/goldset/cases?${q.toString()}`,
  );
}

export interface GoldsetCaseEntry {
  reviewer_a_name: string;
  reviewer_a_label: string;
  reviewer_a_rationale: string;
  reviewer_b_name: string;
  reviewer_b_label: string;
  reviewer_b_rationale: string;
  adjudication_status: string;
  final_label: string;
  final_rationale: string;
  reason_code: string;
  evidence_sufficiency: string;
  difficulty: string;
}

export type GoldsetRole = "reviewer_1" | "reviewer_2" | "adjudicator";

export interface GoldsetCaseDetail {
  case_id: string;
  violation_type: string;
  current_ground_truth: "TP" | "FP" | null;
  agreement: "agree" | "disagree" | "single_review";
  role: GoldsetRole | null;
  owned_fields: string[];
  adjudicator_ready: boolean;
  rotation: string[] | null;
  rule_text: string | null;
  reason_options: string[];
  entry: GoldsetCaseEntry;
}

export function getGoldsetCase(caseId: string, reviewer: string): Promise<GoldsetCaseDetail> {
  return apiGet<GoldsetCaseDetail>(
    `/goldset/cases/${encodeURIComponent(caseId)}?reviewer=${encodeURIComponent(reviewer)}`,
  );
}

export function goldsetCaseImagePath(caseId: string, crop: boolean): string {
  return `/goldset/cases/${encodeURIComponent(caseId)}/image?crop=${crop}`;
}

export function annotateGoldsetCase(
  caseId: string,
  reviewer: string,
  updates: Record<string, string>,
): Promise<{ entry: GoldsetCaseEntry; role: GoldsetRole | null; synced_files: string[] }> {
  return apiPost<{ entry: GoldsetCaseEntry; role: GoldsetRole | null; synced_files: string[] }>(
    `/goldset/cases/${encodeURIComponent(caseId)}/annotate`,
    { reviewer, updates },
  );
}

export function syncGoldsetRationale(): Promise<{ updated_files: string[] }> {
  return apiPost<{ updated_files: string[] }>("/goldset/sync-rationale");
}

export function freezeGoldsetV1(
  fewShotManifestPath: string,
): Promise<{ count: number; output_path: string; warnings: string[] }> {
  return apiPost<{ count: number; output_path: string; warnings: string[] }>("/goldset/freeze-v1", {
    few_shot_manifest_path: fewShotManifestPath,
  });
}

export function freezeGoldsetV2(
  fewShotManifestPath: string,
): Promise<{ count: number; output_path: string; warnings: string[] }> {
  return apiPost<{ count: number; output_path: string; warnings: string[] }>("/goldset/freeze-v2", {
    few_shot_manifest_path: fewShotManifestPath,
  });
}

// ---------------------------------------------------------------------------
// Evaluate escalations (/escalations) -- the two human-decision queues.
//
// Both are views over the Track1 audit log, not their own work list, so a
// decided case simply stops matching the default `include_reviewed=false`
// filter. That is the whole "advance to the next case" mechanism: save,
// refetch, and the finished case is gone from `items`.
//
// This screen collects a verdict OR a box and nothing else -- no rationale,
// reason code, evidence rating or difficulty (see CONTRACT.md's own note).
// It feeds CV retraining labels; it is not a grade of the VLM's reasoning,
// so there is deliberately no reviewer rotation, role gating or
// adjudication here the way /goldset has. `reviewer` is provenance only and
// is never validated against a roster.
// ---------------------------------------------------------------------------

/**
 * A box as `[x1, y1, x2, y2]`, in whole pixels of SOME image.
 *
 * ### Which image? Read this before touching the box queue.
 *
 * There are two coordinate spaces on this screen and they are NOT
 * interchangeable:
 *
 *  - **Frame pixels** -- the original 1920x1080-ish alarm frame. This is
 *    what the audit log stores and what `EscalationItem.person_box` /
 *    `.helmet_box` are in. It is also the canonical storage space: the
 *    server keeps every review's box here.
 *  - **Crop pixels** -- the region the judge was actually shown
 *    (`EscalationItem.crop_box` says where it sits in the frame), which is
 *    what `GET .../image` serves by default and therefore what the reviewer
 *    sees and draws on. `EscalationItem.person_box_in_crop` /
 *    `.helmet_box_in_crop` are in this space.
 *
 * **`POST /escalations/cases/{id}/box` takes CROP pixels.** The reviewer
 * draws on the crop, so the client sends exactly what it measured on the
 * served image and the server maps it back to frame pixels itself (the same
 * inverse the stage-2 detector's own box goes through). Do not convert on
 * the way out. See `setEscalationBox`.
 *
 * The conversion between the two spaces is a pure translation by the crop
 * origin, never a scale, because the server crops without resizing -- a
 * pixel of the served crop IS a pixel of the frame, offset by
 * `crop_box[0]`/`crop_box[1]`.
 */
export type PixelBox = [number, number, number, number];

export type EscalationQueueName = "verdict" | "box";

/** What a human recorded for one case, merged per alarm: a `verdict` entry
 * and a `helmet_box_status` entry land in the same sidecar record, so a box
 * decision never wipes a verdict recorded earlier on the same case. */
export interface EscalationReview {
  verdict?: string;
  helmet_box_status?: string;
  /** The recorded box in FRAME pixels -- canonical storage, because the crop
   * is a review-time rendering decision that would go stale if crop padding
   * were ever retuned. */
  helmet_box?: number[] | null;
  /** The same box in CROP pixels: exactly what the reviewer drew and sent.
   * This, not `helmet_box`, is what the drawing tool must be re-seeded with
   * when revisiting a decided case. Absent on reviews saved before the crop
   * review flow existed -- see `reviewedBox` in `EscalationsPage`. */
  helmet_box_in_crop?: number[] | null;
  reviewed_by?: string;
  reviewed_at?: string;
}

export interface EscalationItem {
  alarm_id: string;
  queue: EscalationQueueName;
  hazard_type: string;
  violation_type: string | null;
  timestamp: string | null;
  image_key: string | null;
  image_width: number | null;
  image_height: number | null;
  /** FRAME pixels. Only useful in the full-frame context view -- the review
   * surface is the crop, so the overlays there use the `_in_crop` pair. */
  person_box: number[] | null;
  helmet_box: number[] | null;
  /** The region the judge was shown, in FRAME pixels, and the person/helmet
   * boxes re-expressed in that crop's own pixels.
   *
   * All three are null on rows judged before the crop region was recorded.
   * A null `crop_box` means the case CANNOT be boxed: the server has no way
   * to map a crop-pixel box back to the frame and 400s on the attempt, and
   * `GET .../image?crop=true` quietly serves the whole frame instead. The
   * UI must disable box saving for those rather than let a reviewer draw
   * into an error. */
  crop_box: number[] | null;
  person_box_in_crop: number[] | null;
  helmet_box_in_crop: number[] | null;
  helmet_box_source: HelmetBoxSource;
  helmet_box_confidence: number | null;
  /** What the models decided, shown as context. Never editable here -- a
   * human correction is recorded separately, in the review sidecar. */
  judge_outcome: string | null;
  judge_reason: string | null;
  judge_reasoning: string | null;
  judge_confidence: number | null;
  judge_model: string | null;
  reviewed: boolean;
  review: EscalationReview;
  /** This case is in the HELMET BOX queue because an audit corrected it to
   * `no_violation` — a human saying the helmet IS worn on a case the judge
   * called a violation. The sheet has to say so: a VIOLATION stamp beside a
   * request to draw the helmet reads as a contradiction otherwise. */
  requeued_by_audit?: boolean;
  audit_corrected_verdict?: string | null;
  /** EVERY CROP OF THIS IMAGE, so a reviewer can strike one out while
   * resolving the escalation. The escalation queues collapse a frame's crops
   * to one item (`routers/escalations._one_row_per_image`) and a reviewer
   * cannot strike out a crop they were never shown.
   *
   * EMPTY for a live alarm, which is one crop and has no strip. */
  candidates?: EscalationCandidate[];
  /** WHICH OF THOSE CROPS STILL OWE A HUMAN AN ANSWER, by candidate id.
   *
   * NOT the same question as `candidates[].verdict`, and the strip must not
   * draw them alike: this is what is still OWED, that is what was ANSWERED.
   * A crop absent from here with a null verdict is one the FRAME's own
   * verdict covers, which is a resolved crop nobody singled out.
   *
   * Empty for a live alarm (one crop, no strip) and empty once every hedge
   * on the frame is answered, by either kind of answer. Computed server-side
   * against the STORED review, so it does not move as a reviewer fills the
   * form in -- the screen has to reconcile its own unsaved answers itself. */
  awaiting_verdict?: string[];
}

/** One crop carried on an escalation queue item.
 *
 * ### `alarm_id` HERE IS THE CROP, NOT THE FRAME.
 * It is what `escalationCaseImagePath` serves a crop by; the enclosing
 * `EscalationItem.alarm_id` is the FRAME key the decision is filed under.
 * Swapping the two renders the same picture under every crop in the strip,
 * with nothing to notice. */
export interface EscalationCandidate {
  alarm_id: string;
  /** The key a strike-out is filed under, and the only id
   * `POST .../verdict`'s `excluded_people` accepts. */
  candidate_id: string | null;
  person_index: number | null;
  /** FRAME pixels, like `crop_box` -- shifted by the crop origin to draw over
   * the served crop. */
  person_box: number[] | null;
  crop_box: number[] | null;
  judge_outcome: string | null;
  judge_reason: string | null;
  judge_reasoning: string | null;
  /** Already struck out on file, from the audit sidecar. */
  excluded: boolean;
  /** WHAT A HUMAN ANSWERED ABOUT THIS CROP, and null when nobody has --
   * including on a crop the frame's own verdict covers, because the two are
   * different claims. Stored in the review sidecar under `crop_verdicts` and
   * read back through `escalations.crop_verdict`; it beats the frame answer
   * wherever both apply, and it is what gives an escalated crop a training
   * label of its own. */
  verdict?: string | null;
}

export interface AuditBlocker {
  code: string;
  message: string;
}

export interface AuditReadiness {
  ready: boolean;
  /** Why auditing is not possible yet, in the server's own words. */
  blockers: AuditBlocker[];
  counts: {
    /** Settled by the pipeline with no human involved -- the cases with the
     * least oversight, and so the ones most worth auditing. */
    auto_settled: number;
    settled_by_human: number;
    labelled_by_human: number;
    /** Settled as cannot_determine / no_helmet_visible: finished work that
     * deliberately yields no label. */
    excluded_by_reviewer: number;
    exportable: number;
    exported: number;
    audited: number;
  };
  /** Deterministic share of exportable cases the audit will sample. */
  sample_rate: number;
  audit_target: number;
}

export async function getAuditReadiness(): Promise<AuditReadiness> {
  return apiGet<AuditReadiness>("/escalations/audit/readiness");
}

// ---------------------------------------------------------------------------
// The audit queue (/escalations/audit/queue, /escalations/audit/cases/{id})
//
// An audit spot-checks cases that are already SETTLED. It is not a third
// review queue: nothing here is waiting on a decision before it can move on,
// and what gets recorded is a judgement ABOUT a settled decision rather than
// the decision itself -- which is why the server keeps it in its own sidecar
// instead of merging it into the escalation reviews.
//
// It advances the same way the escalation queues do: an audited case stops
// matching the default `include_audited=false` filter, so refetching after a
// save IS the advance.
//
// Two slices, measuring two different things -- auditing a `pipeline`-settled
// case checks the model (nobody ever looked at it), auditing a `human`-settled
// one checks the reviewer.
// ---------------------------------------------------------------------------

/** Who had the last word on a case before the audit. */
export type AuditSettledBy = "pipeline" | "human";

/** The queue filter: `all`, plus the two slices. */
export type AuditSettledByFilter = "all" | AuditSettledBy;

/** One recorded audit decision. An empty object (`{}`) on a case nobody has
 * audited yet, so every field is optional. */
export interface AuditRecord {
  decision?: string;
  corrected_verdict?: string | null;
  note?: string;
  audited_by?: string;
  audited_at?: string;
  /** The verdict this decision was recorded AGAINST -- what an `agree`
   * agreed with. The server writes it (`audit.record_audit`) and returns the
   * whole entry, and it is what makes a decision retirable: once the settled
   * verdict changes, the entry is stale and stops counting. Absent on entries
   * written before the field existed. */
  audited_verdict?: string | null;
  /** Candidate ids of crops the auditor struck out of the training data.
   * Independent of the verdict: the verdict says what the PICTURE was, this
   * says which of its boxes are not worth training on. */
  excluded_people?: string[];
}

/** Counts and the agreement rate for one slice of the auditable population.
 *
 * `agreement_rate` is `null`, NOT 0, when nothing in the slice has been
 * audited -- "nothing checked yet" and "the auditor disagreed with every
 * single case" are very different claims and must never render the same way.
 *
 * `sample_is_thin` is the server's own `audited < min_samples` check. Read it
 * rather than recomputing: `AuditSummary.min_samples` is reported precisely so
 * there is one source of truth for the threshold. */
export interface AuditBucket {
  auditable: number;
  audited: number;
  pending: number;
  /** The part of `pending` this screen cannot hand you.
   *
   * A blocked case is waiting on somebody to answer a crop the judge hedged
   * on, over on the escalations screen. It is not a rejection and not a
   * verdict, it is work queued behind someone else. `pending - blocked` is
   * what the audit queue returns, so a badge that shows `pending` alone
   * promises rows the list will not hold.
   *
   * Optional: a server older than this field sends nothing, and a count that
   * reads as absent must not render as zero blocked. */
  blocked?: number;
  agreed: number;
  /** "The verdict is right, the box is not." Its own bucket, in neither
   * `agreed` nor `disagreed`: it endorses the verdict (so the confusion
   * matrix still scores the judge as correct) while rejecting the label, so
   * counting it as agreement would hide a bad box and counting it as
   * disagreement would blame the judge for a verdict it got right. */
  box_rejected: number;
  disagreed: number;
  /** Decisions retired because the verdict they were about changed. */
  stale: number;
  /** Settled and audited in one act by the same person (a reviewer resolving
   * an escalation). Counted as audited, held out of the rate: agreeing with
   * the verdict you yourself just gave is tautological. */
  self_settled: number;
  /** Images whose every crop the auditor struck out. Counted as audited and
   * held out of the rate for a different reason from the two above: the case
   * is finished, but there is no verdict left to have agreed or disagreed
   * with. See `NO_VALID_CROPS` in `lib/caseVerdict.ts`. */
  no_valid_crops: number;
  /** Audited, but ENDORSING A VERDICT THE CASE DOES NOT HAVE: every crop
   * still in the image is one the judge hedged on, so `settled_verdict` is
   * `null` and an `agree` agreed with nothing. Its own bucket beside
   * `no_valid_crops` and out of the rate for the same reason -- a human has
   * finished with the case, and there was no verdict to match. Counting these
   * as agreement put 100% on the page beside n=0. */
  no_verdict: number;
  /** `agreed / audited`. Deliberately strict: a `box_rejected` case is not a
   * full endorsement of what was settled, even though the verdict stood.
   *
   * The three buckets above are OUT of the denominator, so
   * `agreed + disagreed + box_rejected` does not reach `audited` whenever any
   * of them is non-zero -- which is why a reader is told about them rather
   * than left to find the gap. */
  agreement_rate: number | null;
  sample_is_thin: boolean;
}

/** The same shape on `GET /escalations/audit/queue` and on
 * `GET /metrics/rolling`'s `audit` block -- one summary, two readers. */
export interface AuditSummary {
  /** A cap on how much is in front of a person at once, not a sample size:
   * audited cases drop out, so finishing some lets more in. */
  limit: number;
  /** Below this many audited cases an agreement rate is not worth reading (at
   * n=3 one disagreement moves it 33 points). Server-reported so the UI never
   * hardcodes it. */
  min_samples: number;
  overall: AuditBucket;
  pipeline_settled: AuditBucket;
  human_settled: AuditBucket;
  /** Cases that reached the training export -- a CROSS-CUT of the two slices
   * above, not a third slice: it overlaps both. An audit checks what was
   * PREPARED, so this is the slice whose findings say something about the
   * dataset itself. */
  exported: AuditBucket;
  /** `agree` / `disagree` / `box_wrong`. Read from here rather than
   * hardcoded. */
  decisions: string[];
  /** `violation` / `no_violation` / `cannot_determine` -- the same vocabulary
   * as an escalation review, so nothing downstream learns a second one. */
  corrected_verdicts: string[];
  /** Precision/recall/F1/accuracy for the JUDGE, scored against audits
   * instead of a gold set -- see `AuditConfusion`. */
  confusion: AuditConfusion;
}

/** One confusion matrix. Same field names and same meanings as the offline
 * eval's, deliberately: an offline eval scores the judge against gold labels,
 * a live alarm has none, so the auditor's confirmation becomes the ground
 * truth and the SAME matrix falls out of it. Two definitions of precision in
 * one app would be worse than none. */
export interface ConfusionBlock {
  /** Audited cases this block rests on. */
  n: number;
  sample_is_thin: boolean;
  counts: {
    true_positive: number;
    false_negative: number;
    false_positive: number;
    true_negative: number;
    uncertain: number;
  };
  precision: number;
  recall: number;
  f1: number;
  accuracy: number;
  /** Accuracy's denominator: escalations are excluded, so hedging cannot
   * drag accuracy down the way a wrong answer does. */
  decided_total: number;
  uncertain_rate: number;
  /** What really WAS a violation, however the judge answered -- the split
   * that says where the errors are, which one accuracy number cannot. */
  ground_truth_violation_total: number;
  ground_truth_no_violation_total: number;
  /** Keyed `hold` / `cleared` / `escalate` -- the server's historical names,
   * shown as VIOLATION / NO VIOLATION / ESCALATE. See
   * `VerdictDistribution`. */
  violation_verdict_distribution: Record<string, number>;
  no_violation_verdict_distribution: Record<string, number>;
}

/**
 * WHERE THE JUDGE FIRED, PER CROP -- and deliberately not a second accuracy.
 *
 * ### This is attribution. There is no per-crop human ground truth.
 * An auditor records ONE verdict for the picture, and `training_export`'s own
 * rule is that on a violation image only the crop that FIRED inherits the
 * label -- pushing the image's verdict down onto its crops would label every
 * bystander in every violation frame. So the only per-crop answer a human
 * ever gives is KEEP or STRIKE OUT, and the only question these counts can
 * answer is: did the judge fire on a crop the auditor kept, or on one they
 * struck out?
 *
 * ### There is no accuracy, precision, recall or F1 here, and none may be
 * derived. "The judge cleared a crop the auditor kept" is the ORDINARY case,
 * not an error: a rate over these counts renders as a judge failure rate of
 * roughly 10% and gets read as a defect. `fired_kept_rate` is the only rate,
 * and only because its denominator (`fired_total`, the crops the judge fired
 * on) is a real population.
 *
 * ### Never summed or visually merged with the blocks above. Those are in
 * IMAGES and are what the 85% shadow-mode target is read off
 * (`CONTRACT.md:134`); this is in CROPS and answers a different question.
 * Both units are labelled on screen for that reason.
 */
export interface CropAttribution {
  unit: "crops";
  /** Crops this block rests on, struck-out ones INCLUDED -- they are its
   * negative class, and holding them out would leave the headline question
   * unanswerable. */
  n: number;
  excluded_crops: number;
  /** The server's own `n < min_samples` check. Read it rather than
   * recomputing: the same human decisions produce ~4x more crops than frames,
   * so the floor here is 4x the image floor and a crop `n` would otherwise
   * clear a frame-tuned floor on identical evidence. */
  sample_is_thin: boolean;
  min_samples: number;
  /** Named for what they are rather than TP/FP/FN/TN: "cleared a crop the
   * auditor kept" is not a false negative in any sense a reader of that word
   * would take.
   *
   * `escalated` NOW CARRIES BOTH HALVES OF ITS SPLIT, and two invariants are
   * asserted server-side: `escalated_on_kept + escalated_on_struck ==
   * escalated`, and `excluded_crops == fired_on_struck + cleared_on_struck +
   * escalated_on_struck`. The second is what makes the struck column an
   * accounting a reader can check against `excluded_crops`. */
  counts: {
    fired_on_kept: number;
    fired_on_struck: number;
    cleared_on_kept: number;
    cleared_on_struck: number;
    escalated: number;
    escalated_on_kept: number;
    escalated_on_struck: number;
  };
  /** `fired_on_kept + fired_on_struck` -- the rate's denominator. */
  fired_total: number;
  /** `null`, never 0, when the judge fired on nothing: "it never fired" and
   * "everything it fired on was junk" are opposite claims. */
  fired_kept_rate: number | null;
}

export interface AuditConfusion {
  overall: ConfusionBlock;
  per_hazard_type: Record<string, ConfusionBlock>;
  min_samples: number;
  /** The unit of `overall` and `per_hazard_type`, on the response rather than
   * in a reader's head. `per_crop` is in another one and the two must never
   * be summed. */
  unit: "images";
  per_crop: CropAttribution;
}

export interface AuditQueueItem {
  alarm_id: string;
  settled_by: AuditSettledBy;
  hazard_type: string;
  timestamp: string | null;
  image_key: string | null;
  /** FRAME pixels, or null on a row judged before the crop region was
   * recorded. Also says whether the image on screen is a crop at all (see
   * `escalationCaseImagePath`'s caveat), which decides whether the two boxes
   * below need the crop origin subtracted before they are drawn. */
  crop_box: number[] | null;
  /** Both in FRAME pixels, like `crop_box`. Drawn over the served crop by
   * subtracting `crop_box[0]`/`crop_box[1]` -- the crop is not resized, so
   * there is no scale factor. The helmet box is the evidence the judge acted
   * on: asking someone whether a verdict was right while not showing them
   * what it was based on is asking them to check work they have not seen. */
  helmet_box: number[] | null;
  person_box: number[] | null;
  /** THE THING BEING AUDITED: the verdict that stands on this case, in the
   * review vocabulary (`violation` / `no_violation` / `cannot_determine`), so
   * an auditor compares like with like rather than against a raw CONFIRM_TP.
   * A human's decision wins over the judge's, because that is what settling
   * means. */
  settled_verdict: string | null;
  judge_outcome: string | null;
  judge_reason: string | null;
  judge_reasoning: string | null;
  judge_model: string | null;
  helmet_box_source: HelmetBoxSource;
  helmet_box_confidence: number | null;
  /** Who settled it, when `settled_by` is `human`. */
  reviewed_by: string | null;
  /** Whether this case reached the training export. Auditable either way, but
   * an unexported case's findings say nothing about the dataset. */
  exported: boolean;
  audited: boolean;
  /** A decision exists on this case, but the verdict it was about has since
   * changed, so it no longer counts and the case is back in the queue.
   * `audited` is false alongside it -- a stale decision is not an audit. */
  stale_audit: boolean;
  audit: AuditRecord;
  /** What this person is still waiting on, when they are carried into a case
   * without being settled: `"verdict"` for a judge that hedged. `null` for
   * everyone else. They are SHOWN rather than hidden -- an escalation used to
   * read on screen as a person nobody had judged. */
  awaiting?: string | null;
  /** The helmet-CV emulation's own answer for this frame, `{}` for an
   * ordinary alarm. Read fields off the router, never guess them. */
  cv?: Record<string, unknown>;
  /** Where this row came from: `{source: "live_cv" | "offline_gold", ...}`,
   * `{}` for rows written before lineage was recorded. */
  lineage?: Record<string, unknown>;
}

/** ONE IMAGE, with every person the judge was asked about nested in it.
 *
 * The unit of an audit (CEO, 1 Sep: "All the crops from 1 image should be
 * part of 1 audit"). A live alarm and a gold case are one person per image,
 * so their case is a group of one and its id is still the alarm id. */
export interface AuditCase {
  case_id: string;
  people: AuditQueueItem[];
  settled_by: AuditSettledBy;
  hazard_type: string;
  timestamp: string | null;
  /** The image's verdict: the OR of the people the auditor KEPT. ANY person
   * in violation makes the image a violation, and a struck-out crop is out of
   * the population that runs over. This is what the decision is about.
   *
   * Can be `"no_valid_crops"` (see `lib/caseVerdict.ts`) when every crop was
   * struck out. That is NOT a clearance and must never render as one.
   *
   * Can be `null` when every crop still in the image is one the judge hedged
   * on: the picture has no verdict YET, which is a third thing again -- crops
   * remain, and nobody has spoken. A case in that state still arrives
   * `audited`, so anything switching on this string has to carry the null. */
  settled_verdict: string | null;
  judge_outcome: string | null;
  /** The words behind that outcome, from THE CROP THAT PRODUCED IT — never
   * `people[0]`. The image's outcome is an OR across its people, so on a
   * violation image the answer belongs to whoever fired, rarely person 1.
   * Reading `people[0]` here showed "clearly wearing a white helmet" under a
   * VIOLATION stamp on 93% of the pilot's violation frames. */
  judge_reason: string | null;
  judge_reasoning: string | null;
  judge_model: string | null;
  /** Which person in `people` those three came from. */
  deciding_person: number;
  /** `live_cv`, `offline_gold`, or `""` for a row written before lineage. */
  source: string;
  /** False when some of this image's people were never judged -- a failed
   * call, or a live feed stopped part way. Their absence would otherwise
   * read as "nobody else was in violation". */
  complete: boolean;
  people_expected: number;
  /** How many crops the first-pass person gate discarded as holding no
   * person. "12 boxes, 2 people" is a fact about the detector. */
  people_gated: number;
  /** People in this image the judge declined to answer about. Part of what
   * the auditor is looking at, not a gap in it. */
  people_awaiting: number;
  /** People an auditor struck out of this image, ON FILE. They stay in
   * `people` -- the strike is reversible and the auditor has to see what they
   * struck -- but they are out of the OR that produced `settled_verdict`.
   *
   * The server's saved count, so it is stale the moment a reviewer touches a
   * checkbox. For the working count, count the form's own strike list; see
   * `lib/caseVerdict.ts`. */
  people_excluded: number;
  exported: boolean;
  audited: boolean;
  stale_audit: boolean;
  audit: AuditRecord;
}

export interface AuditQueueResponse {
  settled_by: AuditSettledByFilter;
  /** OLDEST first, unlike the escalation queues (newest-first): oldest-first
   * keeps membership stable, so a newly settled case joins the end rather than
   * displacing something a person is part-way through. */
  /** ONE ENTRY PER IMAGE. What the screen renders and what a decision is
   * recorded against. */
  cases: AuditCase[];
  /** The same queue flattened to one entry per crop, for a caller reading
   * the ledger rather than reviewing. NOT a second queue: a decision is
   * recorded against a `case_id`, never against a person. */
  items: AuditQueueItem[];
  summary: AuditSummary;
}

/** ONE case by id. The live feed's results table opens the alarm it just
 * judged without pulling the whole queue to find it. */
export function getAuditCase(caseId: string): Promise<AuditCase> {
  return apiGet<AuditCase>(`/escalations/audit/cases/${encodeURIComponent(caseId)}`);
}

export function getAuditQueue(
  settledBy: AuditSettledByFilter,
  includeAudited: boolean,
): Promise<AuditQueueResponse> {
  const q = new URLSearchParams({
    settled_by: settledBy,
    include_audited: String(includeAudited),
  });
  return apiGet<AuditQueueResponse>(`/escalations/audit/queue?${q}`);
}

/**
 * WHAT AN IMAGE WOULD SETTLE TO UNDER A SET OF STRIKE-OUTS.
 *
 * ### THE authority for what the crop strip displays. Do not re-derive it.
 * The precedence that turns a set of crops into an image's verdict is written
 * out at five server-side sites, and `audit.included_people` exists so those
 * cannot drift. A sixth copy in the browser -- which is what this endpoint
 * replaced -- is one that no test on either side of the wire can catch
 * disagreeing, so there is no client-side mirror and no fallback to one: a
 * fallback that disagrees is exactly the drift being removed.
 *
 * Answered through the same `audit.case_outcome` call the queue and
 * `recordAudit` both go through, so a preview and the save that follows it
 * cannot give different answers.
 */
export interface AuditPreview {
  case_id: string;
  /** `"violation"` / `"no_violation"` / `"cannot_determine"` /
   * `"no_valid_crops"` / `null`.
   *
   * `null` IS A STATE, not an absence. It means every crop still in the image
   * is one the judge hedged on, so nobody has spoken for the picture yet --
   * NOT `no_valid_crops` (crops remain) and never a clearance.
   * `people_awaiting_verdict` is what tells it apart from any other null. */
  settled_verdict: string | null;
  /** The judge's own outcome OR'd over the KEPT crops, `null` when none of
   * them carries a readable answer. */
  judge_outcome: string | null;
  /** Indexes the FULL `people` list a screen renders, never a struck crop --
   * so the judge's words on the sheet can be read off the crop that is
   * actually deciding the image now. */
  deciding_person: number;
  people_excluded: number;
  /** The ids that would ACTUALLY apply. One naming a crop of another picture
   * is dropped here exactly as the save drops it, so the preview cannot
   * promise something the save will not do. Unreachable from this UI, whose
   * ids all come off the case's own people. */
  excluded_people: string[];
  /** Kept crops the judge hedged on: why `settled_verdict` is `null` when it
   * is. Without it a screen cannot tell "still waiting on a human" from
   * "nothing here was decided". */
  people_awaiting_verdict: number;
}

/**
 * Asks what these strike-outs settle to. Writes nothing; two file reads.
 *
 * `excludedPeople` is THE WHOLE QUESTION, not a delta -- the strikes already
 * recorded on the case play no part -- or "what if I un-tick this one" could
 * not be asked.
 *
 * `404` for a case id that no longer resolves, matching `getAuditCase` and
 * `recordAudit`: the client is looking at something re-run or audited away.
 * It is a signal to RELOAD, not an error to swallow -- inventing a verdict
 * for it would put a confident answer on screen about a case that is gone.
 */
export function previewAuditCase(
  caseId: string,
  excludedPeople: string[],
): Promise<AuditPreview> {
  return apiPost<AuditPreview>(
    `/escalations/audit/cases/${encodeURIComponent(caseId)}/preview`,
    { excluded_people: excludedPeople },
  );
}

/** What saving an audit did to the training set.
 *
 * An audit that only recorded a number would leave a label a human just
 * called wrong sitting in the dataset, so the server acts on it and reports
 * what it did. `action` is `none` / `retracted` / `retracted_and_requeued`;
 * `detail` is the server's own ready-to-read sentence and is absent when
 * nothing happened, which is why the UI shows the sentence rather than
 * writing its own from the action. */
export interface AuditDatasetEffect {
  action: string;
  detail?: string;
  /** Present only when struck-out crops were pulled from the crop dataset.
   * A separate sentence from `detail` because it is a separate act: `detail`
   * is what the VERDICT did, this is what the STRIKE-OUTS did, and an `agree`
   * that strikes a crop changes no verdict at all. */
  excluded_crops?: string;
  /** Present when the full-frame dataset was republished or retracted. */
  frame_dataset?: string;
}

/**
 * Records one audit decision.
 *
 * ### `correctedVerdict` is REQUIRED on `disagree` and REFUSED on `agree`.
 * Both are `400`s server-side, and both are worth blocking client-side so an
 * auditor never hits an avoidable one: a disagreement without the correction
 * gives only an error rate, while the correction also gives a usable label,
 * and a correction attached to an agreement is meaningless. It is omitted from
 * the body entirely rather than sent as `null` when there isn't one.
 *
 * `box_wrong` ("the verdict is right, the helmet box is not") also refuses a
 * correction (`400`): it endorses the verdict and rejects only the label.
 *
 * A value outside `AuditSummary.decisions` / `.corrected_verdicts` is a `422`
 * (whose `detail` is FastAPI's validation array, not a string -- see
 * `apiErrorText` in `AuditPage`), and an alarm that is not a settled,
 * auditable case is a `404`. Neither is reachable from a UI that drives its
 * choices off the summary and its cases off the queue.
 */
/** `excludedPeople` is ALWAYS sent, empty list included: an empty list is how
 * a previous strike-out is cleared, so omitting it when nothing is struck
 * would make un-striking impossible. */
export function recordAudit(
  alarmId: string,
  decision: string,
  correctedVerdict: string | null,
  auditor: string,
  note: string,
  /** The WHOLE strike-out list, or `null` to leave the stored one alone.
   *
   * `null` and `[]` are different requests and the server treats them so:
   * `[]` CLEARS the auditor's strikes. Defaulting this to `[]` is what let a
   * helmet-box review wipe an auditor's work, flip the verdict back and
   * republish a pulled crop -- fixed server-side, and defaulted here to the
   * harmless value so the next caller cannot reintroduce it by omission. */
  excludedPeople: string[] | null = null,
): Promise<{ alarm_id: string; audit: AuditRecord; dataset: AuditDatasetEffect }> {
  return apiPost(`/escalations/audit/cases/${encodeURIComponent(alarmId)}`, {
    decision,
    ...(correctedVerdict ? { corrected_verdict: correctedVerdict } : {}),
    auditor,
    note,
    ...(excludedPeople === null ? {} : { excluded_people: excludedPeople }),
  });
}

/** THE TWO DATASETS. Same class indices, incompatible geometry: `crop` holds
 * one person per image, `frame` holds whole frames with every person in them
 * labelled. Never mix a sample of one with the other's boxes. */
export type DatasetName = "crop" | "frame";

export interface DatasetSampleRow {
  alarm_id: string;
  image_name: string;
  split: string;
  unit: string;
  classes: string[];
  /** Boxes in this sample's label file. `null` only when neither the boxes
   * nor the index's own counts were recorded. */
  box_count: number | null;
  people_labelled?: number | null;
  helmets_labelled?: number | null;
  settled_by?: string | null;
  human_verdict?: string | null;
  batch_id?: string | null;
}

export interface DatasetSamples {
  dataset: DatasetName;
  prefix: string;
  class_names: Record<string, string>;
  count: number;
  splits: Record<string, number>;
  samples: DatasetSampleRow[];
}

/** One sample, with boxes read back off its LABEL FILE -- what a trainer
 * reads, so a label that had drifted from the metadata cannot render as
 * correct. `yolo` is `cx cy w h` normalized to the served image. */
export interface DatasetSample {
  dataset: DatasetName;
  alarm_id: string;
  split: string;
  unit: string;
  image_name: string;
  boxes: { class_id: number; class_name: string; yolo: number[] }[];
  box_sources: string[];
  label_text: string;
  human_verdict?: string | null;
  settled_by?: string | null;
  helmets_disputed?: boolean | null;
}

export function getDatasetSamples(dataset: DatasetName): Promise<DatasetSamples> {
  return apiGet<DatasetSamples>(`/escalations/dataset/samples?dataset=${dataset}`);
}

export function getDatasetSample(dataset: DatasetName, alarmId: string): Promise<DatasetSample> {
  return apiGet<DatasetSample>(
    `/escalations/dataset/samples/${encodeURIComponent(alarmId)}?dataset=${dataset}`,
  );
}

/** Take one sample OUT of the dataset: its image, its label, its metadata, in
 * the bucket. A last resort: republishing the case puts it back, so a reason
 * belongs on the audit screen instead, where it is recorded and moves the
 * metrics. */
export function removeDatasetSample(
  dataset: DatasetName,
  alarmId: string,
): Promise<{ removed: boolean; detail: string; note: string }> {
  return apiDelete(
    `/escalations/dataset/samples/${encodeURIComponent(alarmId)}?dataset=${dataset}`,
  );
}

export function datasetSampleImagePath(dataset: DatasetName, alarmId: string): string {
  return `/escalations/dataset/samples/${encodeURIComponent(alarmId)}/image?dataset=${dataset}`;
}

/** One case as it would land in the YOLO dataset. */
export interface TrainingExportCase {
  alarm_id: string;
  /** The image file this case writes under `images/train/`. */
  image_name: string;
  /** Dataset class ids carried by this case's boxes -- the real dataset's
   * own indices (0 helmet, 2 person), kept whole so exported labels stay
   * compatible with it. Names live in `EXPORT_CLASS_NAMES` on the page: the
   * server does not send a map with this response. */
  classes: number[];
  /** `human` where a reviewer drew the box, `cv` where the detector did --
   * the one thing that says whether a label is a correction or a copy of
   * what the model already believed. */
  box_sources: string[];
  /** Whether a human settled the case or the pipeline settled it alone. */
  settled_by: "human" | "pipeline";
}

/** A settled case that could NOT be exported, in the server's own words.
 * Worth surfacing per case rather than swallowing: an export of 3 out of 13
 * settled cases reads as broken until the other 10 can say why. */
export interface TrainingExportSkip {
  alarm_id: string;
  reason: string;
}

export interface TrainingExportPreview {
  batch_id: string;
  /** Server-side only. A client-supplied destination would be a way to
   * write anywhere in the bucket, so there is nothing to send here. */
  prefix: string;
  case_count: number;
  cases: TrainingExportCase[];
  skipped: TrainingExportSkip[];
}

export interface TrainingExportResult extends TrainingExportPreview {
  /** Cases in the dataset after this run -- cumulative, not this batch. */
  exported_total: number;
  metadata_case_count: number;
}

/** What an export WOULD write, writing nothing. Separate request from the
 * POST on purpose, so the decision is inspectable before a run mutates a
 * shared bucket. */
export function previewTrainingExport(): Promise<TrainingExportPreview> {
  return apiGet<TrainingExportPreview>("/escalations/export/preview");
}

/**
 * Writes the Ultralytics YOLO dataset (images, labels, `data.yaml`,
 * cumulative `metadata.json`, this batch's manifest, and the export index).
 *
 * ### This WRITES TO A SHARED BUCKET, and it is additive.
 * Exporting twice tops the dataset up under stable per-case names rather
 * than replacing it, so a second run is safe but not a no-op -- keep it
 * behind a deliberate press, never a load or a retry.
 *
 * `400` with a `detail` listing the reasons when nothing is exportable, and
 * `502` when the write itself fails; both name the cause and are worth
 * surfacing verbatim.
 */
/** Starts a full publish in the BACKGROUND and returns immediately. A full
 * publish is ~3 storage round trips per case, so 130 cases runs for minutes;
 * poll `getTrainingExportProgress` and show the count. `409` means one is
 * already running for this session. */
export function runTrainingExport(): Promise<{ job_id: string; total: number }> {
  return apiPost<{ job_id: string; total: number }>("/escalations/export");
}

/** Discriminated on `status`, like every other job in this app. `"none"` means
 * no export has been started this session, which is not an error. */
export type TrainingExportProgress =
  | { status: "none" }
  | {
      status: "running" | "done" | "error";
      job_id: string;
      total: number;
      processed: number;
      /** Present once `done`: the same shape the synchronous export used to
       * return directly. */
      summary: TrainingExportResult | null;
      error: string | null;
    };

export function getTrainingExportProgress(): Promise<TrainingExportProgress> {
  return apiGet<TrainingExportProgress>("/escalations/export/progress");
}

/** One frame in the FULL-FRAME dataset. */
export interface FrameExportCase {
  alarm_id: string;
  image_name: string;
  classes: number[];
  box_sources: string[];
  /** People labelled in this frame, and helmets. The counts are the point:
   * a full-frame sample is only usable if every person in it is labelled. */
  people_labelled: number | null;
  helmets_labelled: number | null;
  /** Detector boxes the first-pass person gate emptied and this export
   * therefore did NOT label. A box holding a ship would teach the detector
   * to find ships. */
  boxes_gated: number | null;
}

/** The FULL-FRAME dataset: one sample per image, boxes in frame coordinates,
 * beside the crop dataset under its own prefix in the same bucket.
 *
 * Helmet-CV frames only. A live alarm carries ONE person box for a frame that
 * may hold eight people, so exporting it whole would teach the detector that
 * every other worker is background -- which is the reason the crop dataset
 * exists and still does. */
export interface FrameExportPreview {
  batch_id: string;
  /** Server-side only, like the crop export's. */
  prefix: string;
  case_count: number;
  unit: "full_frame";
  cases: FrameExportCase[];
  skipped: TrainingExportSkip[];
}

export interface FrameExportResult extends FrameExportPreview {
  exported_total: number;
}

/** What the full-frame dataset WOULD contain, writing nothing. */
export function previewFrameExport(): Promise<FrameExportPreview> {
  return apiGet<FrameExportPreview>("/escalations/export/frames/preview");
}

/** Publishes the full-frame dataset in the BACKGROUND. Writes to a shared
 * bucket and is additive, exactly like the crop export -- keep it behind a
 * deliberate press. `409` if one is already running for this session. */
export function runFrameExport(): Promise<{ job_id: string; total: number }> {
  return apiPost<{ job_id: string; total: number }>("/escalations/export/frames");
}

export type FrameExportProgress =
  | { status: "none" }
  | {
      status: "running" | "done" | "error";
      job_id: string;
      total: number;
      processed: number;
      summary: FrameExportResult | null;
      error: string | null;
    };

export function getFrameExportProgress(): Promise<FrameExportProgress> {
  return apiGet<FrameExportProgress>("/escalations/export/frames/progress");
}

export interface EscalationQueueCount {
  pending: number;
  total: number;
  /** Cases in this queue that CANNOT be worked, because the alarm's frame
   * was never stored. Every row judged before frame persistence is in this
   * state, so without this an empty queue is inexplicable: the audit log
   * plainly has rows in it. */
  unreviewable: number;
}

export interface EscalationQueueResponse {
  queue: EscalationQueueName;
  /** Newest-first. */
  items: EscalationItem[];
  /** Pending/total for BOTH queues, so the tab counts stay right without a
   * second request per tab. */
  counts: Record<EscalationQueueName, EscalationQueueCount>;
  /** `violation` / `no_violation` / `cannot_determine`. Read from here
   * rather than hardcoded, so an option the server adds or drops shows up
   * without a frontend change. */
  verdict_options: string[];
  /** `drawn` / `accepted_cv` / `no_helmet_visible`. `accepted_cv` keeps the
   * DETECTOR's box with a human's endorsement and takes no box payload:
   * redrawing a box that is already right is wasted work and worse data.
   * Read from here rather than hardcoded. */
  box_status_options: string[];
  /** `helmet_box.review_confidence` from the server's own config -- the
   * source of truth for `HELMET_BOX_REVIEW_CONFIDENCE`'s local copy. */
  review_confidence: number;
}

export function getEscalationQueue(
  queue: EscalationQueueName,
  includeReviewed: boolean,
): Promise<EscalationQueueResponse> {
  const q = new URLSearchParams({ queue, include_reviewed: String(includeReviewed) });
  return apiGet<EscalationQueueResponse>(`/escalations/queue?${q}`);
}

/**
 * Path for `useApiImage`. UNANNOTATED on purpose -- the box screen draws its
 * own overlay, and a server-burned rectangle would be indistinguishable from
 * the one the reviewer is placing. Searches both queues server-side, so the
 * caller needn't know which one holds this alarm.
 *
 * `crop` defaults to true, matching the server, and that default is the one
 * to keep: the worker is often ~70px tall in a 1920x1080 frame, so the full
 * frame is not a surface anyone can place a helmet box on. The crop is the
 * exact region the judge saw, served WITHOUT resizing -- which is what makes
 * a pixel of it a pixel of the frame plus the crop origin, and what lets the
 * drawn box go back to the server in crop pixels unconverted.
 *
 * `crop=false` is the whole frame, for context only. It is a different
 * coordinate space, so nothing measured on it may be submitted as a box.
 * When the row has no `crop_box`, the server serves the full frame even for
 * `crop=true` -- so "I asked for the crop" is not "I am looking at a crop".
 */
export function escalationCaseImagePath(alarmId: string, crop = true): string {
  const q = new URLSearchParams({ crop: String(crop) });
  return `/escalations/cases/${encodeURIComponent(alarmId)}/image?${q}`;
}

/** Reviewer-facing wording for the review verdict vocabulary -- the one the
 * server uses for an escalation verdict (`verdict_options`) AND for an audit's
 * corrected verdict (`AuditSummary.corrected_verdicts`), which is deliberately
 * the same vocabulary so nothing downstream learns a second one. Shared for the
 * same reason: two screens present these words, and a wording that drifts
 * between them would read as two different sets of choices.
 *
 * A value the server adds later that isn't named here still renders (as its raw
 * value) rather than disappearing from the choice. */
const VERDICT_LABEL: Record<string, string> = {
  violation: "Violation",
  no_violation: "No violation",
  cannot_determine: "Cannot determine",
};

/** Reason codes a reviewer should read as WORDS, not as identifiers.
 *
 * Everything else stays verbatim on purpose -- `helmet_worn` and `no_helmet`
 * are the judge's own vocabulary and a reviewer learns them in one case. This
 * one is different: `no_person_present` is not the judge's answer at all, it
 * is the first-pass person check reporting that the detector's box held no
 * worker, and "no_person_present" beside a NO VIOLATION stamp reads as a
 * missing translation rather than as a finding. */
const REASON_LABEL: Record<string, string> = {
  no_person_present: "No person detected in crop",
};

export function reasonLabel(reason: string | null | undefined): string {
  if (!reason) return "";
  return REASON_LABEL[reason] ?? reason;
}

export function verdictLabel(verdict: string): string {
  return VERDICT_LABEL[verdict] ?? verdict;
}

/** `verdict` must be one of the queue response's own `verdict_options`
 * (`422` otherwise). `cannot_determine` settles the case and clears it from
 * the queue like any other verdict, but is deliberately NOT a label: the
 * server excludes it from the training export, because a case is escalated
 * precisely when the evidence is too thin to settle and a forced binary
 * choice would put a reviewer's guess into CV training data. */
/** `excludedPeople` is ALWAYS sent, empty list included: an empty list is how
 * a previous strike-out is cleared, so omitting it when nothing is struck
 * would make un-striking impossible. Ids must come from the item's own
 * `candidates[].candidate_id`; the server drops anything belonging to another
 * case.
 *
 * There is no second store for it. Resolving an escalation already writes an
 * audit entry, so the strike-outs ride on that and `audit.excluded_people`
 * stays the one reader -- which is also why the audit screen shows a case
 * resolved here with its strikes already in place. */
/** `cropVerdicts` is A WHOLE MAP keyed by `candidates[].candidate_id`, and
 * sending it REPLACES every crop answer stored for the case: `null` leaves
 * them alone, `{}` clears them, exactly the discipline `excludedPeople`
 * carries above. Values must be `verdict_options` members or the server 422s,
 * and ids belonging to another case are dropped server-side.
 *
 * `verdict` stays REQUIRED alongside it. The frame answer is what covers
 * every crop the reviewer did not single out, and it is what clears a frame;
 * per-crop is an addition to it, never a replacement.
 *
 * THE FRAME ANSWER IS WHAT THE AUDIT RECORDS AS GROUND TRUTH. `set_verdict`
 * writes `payload.verdict` into the audit entry, so a frame `no_violation`
 * sent alongside a crop marked `violation` would put a clearance into the
 * confusion matrix for a frame a human had just called a violation. The
 * escalation screen refuses to submit that pair rather than deriving one
 * silently -- see `contradictingCrops` there. */
export function setEscalationVerdict(
  alarmId: string,
  verdict: string,
  reviewer: string,
  /** The WHOLE strike-out list, or `null` to leave the stored one alone.
   * `[]` clears it. See `recordAudit` for why this defaults to `null`. */
  excludedPeople: string[] | null = null,
  cropVerdicts: Record<string, string> | null = null,
): Promise<{
  alarm_id: string;
  review: EscalationReview;
  /** What the save did to the training set, in the server's own words.
   * `excluded_crops` is present only when struck-out crops were pulled. */
  dataset?: { excluded_crops?: string } & Record<string, unknown>;
}> {
  return apiPost(`/escalations/cases/${encodeURIComponent(alarmId)}/verdict`, {
    verdict,
    reviewer,
    ...(excludedPeople === null ? {} : { excluded_people: excludedPeople }),
    ...(cropVerdicts === null ? {} : { crop_verdicts: cropVerdicts }),
  });
}

/**
 * ### `helmetBox` IS IN CROP PIXELS, NOT FRAME PIXELS.
 *
 * Measured on the image `escalationCaseImagePath(id)` served -- i.e. the
 * judged crop, which is what the reviewer drew on. Send it exactly as
 * measured; the server maps it back to frame pixels itself and stores it
 * there. Converting to frame pixels here would double-apply the crop origin
 * and silently record a box in the wrong place.
 *
 * Errors worth surfacing verbatim, because each names the fix:
 *  - a box past the crop's edge -> `400` "...falls outside the WxH crop."
 *  - a row with no recorded `crop_box` -> `400` "no recorded crop region...
 *    Re-run the alarm through Run pipeline to record one." Such a case
 *    cannot be boxed at all, so don't offer the drag in the first place.
 *  - `status: "drawn"` without a box, or a box on any other status -> `400`.
 *  - inverted, negative, or under-2px boxes -> `422`.
 *
 * "No helmet visible" is NOT a verdict change -- it records that no helmet
 * could be found to draw, so the case is excluded from training data rather
 * than exported with a guessed rectangle. It needs no crop region, so it
 * stays available even on a case whose box cannot be drawn.
 */
export function setEscalationBox(
  alarmId: string,
  status: string,
  /** CROP pixels. See the note above -- this is the one easy thing to get
   * wrong on this screen. */
  helmetBoxInCrop: PixelBox | null,
  reviewer: string,
): Promise<{ alarm_id: string; review: EscalationReview }> {
  return apiPost(`/escalations/cases/${encodeURIComponent(alarmId)}/box`, {
    status,
    helmet_box: helmetBoxInCrop,
    reviewer,
  });
}

// ---------------------------------------------------------------------------
// Reset -- clears everything one offline test run produced, so the next
// starts clean. Destructive; see CONTRACT.md's `POST /metrics/reset`.
// ---------------------------------------------------------------------------

export interface ResetPreview {
  judged_cases: number;
  escalation_reviews: number;
  audit_decisions: number;
  alarm_images: number;
  exported_files: number;
  /** The word `resetOfflineState` must send. Server-reported so the UI never
   * hardcodes it. */
  confirmation: string;
  /** What survives -- a reset destroys a test run's output, never a person's
   * input. Listed by the server so this copy has one source of truth. */
  preserved: string[];
}

export function getResetPreview(): Promise<ResetPreview> {
  return apiGet<ResetPreview>("/metrics/reset/preview");
}

export interface ResetResult {
  cleared: {
    judged_cases: number;
    escalation_reviews: number;
    audit_decisions: number;
    alarm_images: number;
    exported_files: number;
  };
  /** Deletion continues past an individual failure rather than aborting
   * halfway into exactly the mixed state a reset exists to prevent. */
  failures: { key: string; error: string }[];
}

export function resetOfflineState(confirmation: string): Promise<ResetResult> {
  return apiPost<ResetResult>("/metrics/reset", { confirm: confirmation });
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

/** `pending` = alarms where nobody has reviewed any candidate yet. The wire
 * expresses this as `reviewed=false`; it does not mean partially reviewed. */
export type HelmetCvReviewedFilter = "all" | "pending";

/** Which frame of the alarm one row came from. `alarm` is the frame production
 * fired on and the only one the CV arm is scored on. */
export type HelmetCvFrameRole = "alarm" | "clip";

/** What the DETECTOR found in one frame. It reaches no verdict.
 *
 * There is no `state`, no `gate_fired` and no `gate_reason` any more: the
 * cascade that produced them was deleted after measurement showed it could
 * not run on this footage (70% of its candidates had no pose, because the
 * workers are ~83px tall). The detector now says where the people are, the
 * VLM decides, and a human scores both.
 *
 * ### NULL UNTIL THE REVIEWER HAS DECIDED.
 * Blindness on this screen is SERVER-SIDE: the whole object is withheld until
 * a review is posted for that frame, so a consumer must handle its absence as
 * the normal pre-decision state rather than fetching it and declining to
 * render it. There is no client-side hiding to defeat. */
export interface HelmetCvCall {
  candidate_id: string;
  cv_run_id: string | null;
  frame_role: HelmetCvFrameRole | null;
  person_index: number | null;
  /** ORIGINAL FRAME PIXELS, `[x1, y1, x2, y2]` -- the frames endpoint serves
   * the whole frame, uncropped, so these need no crop-origin shift. */
  person_box: number[] | null;
  person_confidence: number | null;
  /** Context, not a decision. Nothing pairs a helmet to a head any more. */
  people_in_frame: number | null;
  helmets_in_frame: number | null;
}

/** The judge's call on the same FRAME -- the VLM verification stage. Withheld
 * on the same terms as `cv`.
 *
 * ### THIS TYPE USED TO BE FICTION.
 * It declared `verdict` / `confidence_score` / `rationale`; the server has
 * always sent `outcome` / `reason` / `reasoning`. Nothing matched, so
 * `confidence_score.toFixed(2)` threw and unmounted the panel the instant any
 * review landed — and `tsc`, `npm run build` and `oxlint` all passed, because a
 * hand-written interface is an ASSERTION about the server, not a check of it.
 * Anything added here must be read off `_judge_block` in
 * `routers/helmet_cv.py`, not guessed. */
export interface HelmetCvJudgeCall {
  /** `CONFIRM_TP` | `CONFIRM_FP` | `ESCALATE` — the judge's raw outcome for
   * the frame's representative crop. Prefer `frame_verdict`. */
  outcome: string | null;
  reason: string | null;
  /** The judge's own words, FOR THE CROP THAT DECIDED THE FRAME -- see
   * `deciding_person_index`. Sent as `reasoning`, never `rationale`. */
  reasoning: string | null;
  /** Which person the block above describes. The frame verdict is an OR, so on
   * a violation frame this is whoever fired, rarely person 0. Label the
   * reasoning with it: an explanation about a different worker, shown under
   * the frame's stamp, reads as the model contradicting itself. */
  deciding_person_index: number | null;
  model: string | null;
  settled_verdict: string | null;
  helmet_box: number[] | null;
  helmet_box_source: string | null;
  helmet_box_confidence: number | null;
  crop_box: Record<string, number> | null;
  /** The FRAME's answer, OR'd across its crops by the server: `TP` | `FP` |
   * `uncertain` | null. This is the judge's verdict on the frame. */
  frame_verdict: string | null;
  /** Which crops the judge called a violation. ATTRIBUTION, not the answer —
   * a frame both sides call a violation counts as agreement even if the judge
   * pointed at the wrong worker. */
  fired_candidates: string[];
  /** Per-crop detail behind `frame_verdict`. */
  crops: { candidate_id: string | null; outcome: string | null; reason: string | null }[];
}

/** The reviewer's own decision, plus the server's scoring of the other two
 * against it.
 *
 * ### `cv_outcome` / `judge_outcome` ARE THE SERVER'S, ALWAYS.
 * Never recompute correctness here by comparing `cv.state` to
 * `judge.verdict`: two definitions of correctness in one app is worse than
 * none, and the mapping from a CV state to a violation is pipeline logic this
 * layer does not hold. Equality between the two outcomes is the one thing
 * this screen reads off them (a split), and both sides of it are the
 * server's own figures. */
export interface HelmetCvReview {
  verdict: string;
  reviewer: string | null;
  note: string | null;
  reviewed_at: string | null;
  cv_outcome: string | null;
  judge_outcome: string | null;
}

export interface HelmetCvFrameRow {
  /** The people in this frame. `candidate_id`, `person_index` and `person_box`
   * cross the blind because "is THIS person wearing a helmet?" cannot be asked
   * without saying which person — but the VERDICT is about the frame, so these
   * identify, they do not ask. */
  candidates: {
    candidate_id: string | null;
    person_index: number | null;
    person_box: number[] | null;
    /** ORIGINAL FRAME PIXELS. The candidate image endpoint crops to this box. */
    crop_box: number[] | null;
  }[];
  /** How many people the detector found here. `candidates.length`. */
  people: number;
  frame_name: string;
  /** Ordinal position within this alarm, 0-based. Always present.
   *
   * `frame_index` and `offset_ms` are the TRUE clip position and are withheld
   * until a review exists, because they identify the alarm frame: sampled
   * frames sit on the stride grid and the full-scan alarm frame does not (44%
   * of them, measured). Use `position` for ordering and navigation. */
  position: number;
  frame_index: number | null;
  offset_ms: number | null;
  frame_utc: string | null;
  width: number | null;
  height: number | null;
  cv: HelmetCvCall | null;
  judge: HelmetCvJudgeCall | null;
  review: HelmetCvReview | null;
}

export interface HelmetCvFramesResponse {
  alarm_id: string;
  frames: HelmetCvFrameRow[];
}

/** One alarm's row in the ledger.
 *
 * No site and no camera, deliberately: the extractor writes both as the
 * literal string "unk" for this whole corpus (clip keys carry neither), so a
 * Location column would be a column of one word. The fact is stated once, on
 * the frame sheet, through `formatLocation`. */
export interface HelmetCvAlarmRow {
  alarm_id: string;
  clip_key: string;
  clip_start_utc: string | null;
  frames_total: number;
  /** How many people the detector found across this alarm's REVIEWED frames.
   * Every crop is judged, so there is no "sent to the judge" subset; what
   * matters is how many people the judge was shown at all. */
  people_found: number;
  /** DISTINCT frames, not reviewed candidates: one verdict settles a frame
   * however many people are in it. */
  frames_reviewed: number;
}

export interface HelmetCvAlarmsResponse {
  alarms: HelmetCvAlarmRow[];
}

/** Immutable execution context for one CV scoring pass. The reviewer picks
 * this before opening the ledger; every subsequent request is scoped to it. */
export interface HelmetCvRun {
  cv_run_id: string;
  /** Inference resolution the detector ran at, or null for ultralytics' 640.
   * It replaced the retired min-person-height mode as the parameter worth
   * naming, because it decides how many people the detector finds -- and so
   * caps how many the judge is ever shown. */
  imgsz: number | null;
  detector_weights: string | null;
  created_at: string | null;
  legacy: boolean;
}

export interface HelmetCvRunsResponse {
  runs: HelmetCvRun[];
  default_cv_run_id: string | null;
}

/** One arm's score against the reviewer calls. Every figure is the server's;
 * none is computed in this app. All four are `null` until enough candidates
 * have been audited to make them mean anything -- a rate over an empty review
 * log is not a figure, and a zero would read as a claim. */
export interface HelmetCvArmMetrics {
  precision: number | null;
  recall: number | null;
  f1: number | null;
  accuracy: number | null;
  /** Audited candidates this arm was scored over. */
  n: number;
}

/**
 * Two scorings of the same reviewer calls. The distinction between them is
 * the point of the screen, not a detail of it.
 *
 * `cv` and `pipeline` are the comparison: the gate on its own, then the gate
 * with the judge behind it. `pipeline` MINUS `cv` is what adding the VLM
 * stage buys, and producing that one figure is why this feature exists.
 *
 * `judge` is a DIAGNOSTIC, not the third leg of that comparison. It scores
 * the judge on every sampled candidate, including ones CV never routed, so it
 * flatters the real pipeline: production would never have sent the judge
 * those frames. It earns its place because when `pipeline` disappoints, it is
 * the only thing that says whether the gate or the judge is the cause.
 */
export interface HelmetCvSummary {
  cv_run_id: string;
  frames_reviewed: number;
  alarms_total: number;
  /**
   * PRODUCTION CV's precision -- and precision is ALL it has.
   *
   * `recall`, `f1` and `accuracy` arrive as `null` and that is not missing
   * data. Every alarm in this corpus exists because production CV fired, so
   * CV asserted a violation on all of them and a negative on none: recall has
   * no denominator and accuracy no true negatives. Render the nulls as the
   * dash any unmeasured figure gets. Never substitute a zero, and never
   * compute one here.
   */
  cv: HelmetCvArmMetrics;
  /**
   * The VLM's answer over the same frames, with FULL metrics.
   *
   * It is asked about every frame including ones it clears, so a human
   * "violation" on a frame it cleared is a real false negative and recall has
   * a denominator. Per-crop answers are OR'd up to the frame by the server:
   * any crop it calls a violation makes the frame one.
   *
   * There is no third composed arm. With no gate in front of it the pipeline
   * IS this, and two identical columns under different names would invite
   * someone to subtract them and report the zero as a finding.
   */
  vlm: HelmetCvArmMetrics;
  /** What the VLM would have done to the alarms production raised.
   * `suppressed` is the product: alarms an operator would not have seen. */
  cv_vs_vlm: HelmetCvVsVlm;
  population: string;
  paired_intersection?: HelmetCvPairedIntersection;
}

/** How the two arms landed on the frames production actually alarmed on.
 *
 * There is deliberately no `judge_only` cell. It would mean CV cleared
 * something and the VLM held it, and CV cleared nothing in this corpus -- the
 * corpus IS the alarms it raised. A cell that can never be populated is worse
 * than an absent one, because a reader takes its zero for a finding. */
export interface HelmetCvVsVlm {
  decided_total: number;
  /** Both called it a violation. */
  confirmed: number;
  /** CV raised it, the VLM cleared it. The number this exists to produce. */
  suppressed: number;
  /** The VLM declined to answer. Not a disagreement. */
  judge_escalated: number;
  suppression_rate: number | null;
  /** Alarm frames the judge never finished -- a failed call, or a live feed
   * stopped between people. Held out of every count above, because OR'ing a
   * frame short of its people reads as "no violation". */
  incomplete_frames: number;
  population: string;
}

/** The server's full confusion block for a run on the paired cohort. The UI
 * reads only its headline rates; it never derives them from `counts`. */
export interface HelmetCvConfusionBlock extends HelmetCvArmMetrics {
  sample_is_thin: boolean;
  counts: {
    true_positive: number;
    false_negative: number;
    false_positive: number;
    true_negative: number;
    uncertain: number;
  };
  decided_total: number;
  ground_truth_violation_total: number;
  ground_truth_no_violation_total: number;
}

export interface HelmetCvPairedIntersection {
  ground_truth_ids: string[];
  n: number;
  left: HelmetCvConfusionBlock;
  right: HelmetCvConfusionBlock;
}

export function listHelmetCvRuns(): Promise<HelmetCvRunsResponse> {
  return apiGet<HelmetCvRunsResponse>("/helmet-cv/runs");
}

export function getHelmetCvAlarms(
  cvRunId: string,
  reviewed: HelmetCvReviewedFilter,
  limit: number,
): Promise<HelmetCvAlarmsResponse> {
  const q = new URLSearchParams({
    cv_run_id: cvRunId,
    limit: String(limit),
  });
  if (reviewed === "pending") q.set("reviewed", "false");
  return apiGet<HelmetCvAlarmsResponse>(`/helmet-cv/alarms?${q}`);
}

export function getHelmetCvFrames(
  alarmId: string,
  cvRunId: string,
): Promise<HelmetCvFramesResponse> {
  const q = new URLSearchParams({ cv_run_id: cvRunId });
  return apiGet<HelmetCvFramesResponse>(
    `/helmet-cv/alarms/${encodeURIComponent(alarmId)}/frames?${q}`,
  );
}

/** Path for `useApiImage`. Raw bytes behind the same Bearer token every other
 * `.../image` endpoint needs, so a plain `<img src>` will not authenticate --
 * this goes through `ImageOverlayFrame`/`useApiImage` like every other frame
 * in the app. The WHOLE frame, uncropped: a box drawn over it is already in
 * its coordinate space. */
export function helmetCvFrameImagePath(frameName: string): string {
  return `/helmet-cv/frames/${encodeURIComponent(frameName)}/image`;
}

/** Where inside ONE alarm the run has got to, in the order they happen.
 *
 * The steps are the ones the Live feed panel already describes in prose ("a
 * clip download, a decode to find the clean frame, a detector pass, then one
 * judge call per person"), so the live reading and the standing explanation
 * name the same five things. */
export type LiveFeedStage =
  /** The checkpoint that announces this alarm. State-store I/O, and it lands
   * on this object or on `activity` depending on which side of an alarm
   * boundary the write falls -- it means the same thing in both places, so
   * both label tables carry the same word for it. */
  | "saving"
  /** The ~0.9MB annotated-still fetch, the box extraction off it, and the
   * scene hash. Previously the longest hole in the readout: `in_flight` was
   * set with no stage at all for the whole of it. */
  | "reading_snapshot"
  | "checking_duplicate"
  | "finding_clip"
  | "extracting"
  | "detecting_people"
  | "judging";

/** What the run is doing when it is between alarms rather than inside one.
 *
 * The ones named here are the ones this build has words for. The field that
 * carries them is deliberately wider (see `LiveFeedState.activity`), so this
 * union's job is to keep the on-screen label table exhaustive, not to bound
 * what the server may send. */
export type LiveFeedActivityStage = "loading_detector" | "listing" | "waiting" | "saving";

export interface LiveFeedState {
  status: "none" | "running" | "stopped" | "error";
  run_id?: string;
  watermark?: string | null;
  /** Set while the feed is running but cannot proceed. `budget` means the
   * caps stopped it; `paused_detail` says in calls exactly why. A run whose
   * cap is below one alarm's cost skips every alarm forever, and "running"
   * on its own reads as working. */
  paused_reason?: string;
  paused_detail?: string;
  /** Where this run started reading: an ISO time, or null for "everything
   * from the oldest unprocessed alarm". */
  start_on_or_after?: string | null;
  /** Which folder drove this run. Always `vendor_cv` since 2 Sep; a state
   * written before that may say `detector`, and its watermark points into
   * `raw_video/` rather than `snapshot/`. */
  box_source?: "detector" | "vendor_cv";
  /** Which examples this run judged against. */
  few_shot_source?: "fixed" | "sampled";
  /** What a sampled run's prior was built from, when it was sampled. */
  few_shot_sample?: {
    examples: number;
    crops_written: number;
    audited_used: { TP: number; FP: number };
    fixed_used: { TP: number; FP: number };
  } | null;
  calls?: number;
  daily_calls?: number;
  per_run_cap?: number;
  daily_cap?: number;
  /** Re-alarms caught on the LISTING's ETag: byte-identical to an image this
   * run already consumed, so they cost nothing to skip -- no download, no
   * decode, no judge call. Measured at zero on a 40-alarm sample of the
   * production bucket, so this is expected to read 0 and is only worth the
   * screen's space when it does not. */
  duplicates_skipped?: number;
  /** Alarms the NEAR-duplicate check caught before extraction: not the same
   * bytes, but the same scene and the same boxes as one already judged. These
   * cost a clip download and a decode before anything noticed, which is why
   * the backend keeps them separate from `duplicates_skipped` rather than
   * summing the two.
   *
   * Whether these were actually skipped is NOT inferable from this number --
   * read `pre_extraction_acts`. The same count means "skipped" on one run and
   * "would have been skipped" on another, and the screen has to say which. */
  pre_extraction_duplicates?: number;
  /** Whether a pre-extraction duplicate on THIS run is actually skipped
   * rather than only recorded. The server's own answer, because the screen
   * cannot infer it: it depends on the run's box source AND on the backend
   * constant `PREFILTER_SKIPS_EXTRACTION`, and the moment that constant flips
   * a detector-mode run starts genuinely skipping. A screen that had reasoned
   * "detector means shadow" would then report "found ... nothing was skipped"
   * while alarms were being skipped and judge calls saved.
   *
   * Absent on any state written before the field existed. The fallback at its
   * use site reproduces what was true for every such state. */
  pre_extraction_acts?: boolean;
  processed_keys?: Record<string, { status: string; calls: number; error?: string }>;
  /** Alarms that failed, oldest first, CARRIED ACROSS RUNS -- a failure is a
   * durable record of an alarm nobody judged. `run_id` says which run
   * produced each one; entries written before that field existed have none. */
  failures?: {
    clip_key: string | null;
    error: string;
    run_id?: string;
    at?: string;
    /** How many times this clip has been tried. At 3 the feed steps over it
     * -- see `skipped_keys`. */
    attempts?: number;
  }[];
  /** The alarm being worked on RIGHT NOW, written before the work starts.
   * One alarm is a clip download, a decode, a detector pass and one judge
   * call per person -- minutes, during which no other field moves. Without
   * this a feed working correctly through its first alarm is indistinguishable
   * from one doing nothing.
   *
   * IT SURVIVES A FAILURE, deliberately. If the poll raises mid-alarm the
   * backend leaves this set and puts `error` in `status`, because it is the
   * only record of which alarm killed the run. So "present" does NOT mean
   * "in progress": every live reading off this object is gated on
   * `status === "running"`, and a non-running state renders it as the place
   * the run stopped. A ticking timer over a dead worker is the same lie this
   * whole field exists to prevent, pointed the other way. */
  in_flight?: {
    clip_key: string;
    alarm_id: string;
    /** When THIS ALARM started. Stamped per alarm; it used to carry the
     * poll's own start time, so every alarm in one poll shared a value that
     * could be an hour old by the time the last of them was read. */
    since: string;
    /** Where inside the alarm the work has got to. In-memory on the worker
     * and NEVER PERSISTED, deliberately: writing it would cost an S3 round
     * trip per stage on a state object heading for ~53 MB. It rides the
     * shared dict the status endpoint returns, so it is absent whenever this
     * session did not start the run, and absent on a server that predates
     * it. A stale value CAN survive incidentally into stored state -- read
     * `owned_by_this_session` before believing one. */
    /** The seven above are the final list. Typed OPEN anyway, like
     * `activity.stage`: this list grew twice while the screen was being
     * built, and nothing here switches exhaustively on it, so an unknown
     * stage drops the line back to the copy that describes an alarm
     * correctly whatever step it is on -- never to a blank. */
    stage?: LiveFeedStage | (string & {});
    /** Crops answered and crops to answer, present only while `stage` is
     * `judging`. This is where the minutes actually go: a 1 Sep measurement
     * put clip download plus detector at 5.1s against ~4 minutes of judging
     * on a 9-crop alarm.
     *
     * `crops_judged` counts crops ANSWERED, which includes the ones whose
     * call failed -- deliberately, so `3 of 9` does not sit still while a
     * call errors out. It is not a success count and must never be labelled
     * as one. */
    crops_judged?: number;
    crops_total?: number;
  } | null;
  /** What the run is doing BETWEEN alarms. MUTUALLY EXCLUSIVE with
   * `in_flight`: the worker pops this when an alarm starts and restores it
   * when the alarm ends, and pops both when the run is not `running`. So this
   * is one slot with two shapes, not two lines that can both be true, and a
   * screen rendering it unconditionally would blank it for the minutes an
   * alarm takes.
   *
   * Still a second field rather than another `in_flight.stage` because the
   * two make different claims: one is about a specific alarm and the other is
   * not, and folding them together would have the screen name an alarm that
   * is not being worked on. Carries the same in-memory-only caveat as
   * `stage`. */
  activity?: {
    /** Typed OPEN, unlike `in_flight.stage`, whose five values are settled.
     * A third value is already coming for the 30-second sleep between polls,
     * and a reading frozen through that sleep is a small reprise of the bug
     * this batch exists to fix. Widened so a server naming a stage this build
     * has no word for is a rendering fallback, not a type error and not a
     * blank line. */
    stage: LiveFeedActivityStage | (string & {});
    /** RESET PER ACTIVITY, not per run: the between-alarms reading restarts
     * its clock after each alarm rather than carrying the run's whole age. */
    since: string;
    /** How far into the listing the scan has read. Absent on any stage with
     * nothing to count. */
    keys_examined?: number;
  } | null;
  /** Whether THIS session started the run, from the status endpoint. The only
   * thing that says an in-memory-only reading (`in_flight.stage`, `activity`)
   * belongs to the live worker rather than to a state read back from storage.
   * Absent on a server that predates it, which is read as "not owned". */
  owned_by_this_session?: boolean;
  /** Clips that failed three times and were stepped over so the alarms behind
   * them are not held hostage. About 14% of this corpus never yields a clean
   * alarm frame, and retrying one of those forever stalls the whole feed. */
  skipped_keys?: Record<string, { error: string; attempts: number }>;
  /** One entry per judged alarm, newest last, capped at the last 200. The
   * durable record is the audit ledger; this is the run's own tail so the
   * screen can report an alarm the moment it lands, the way the offline
   * batch reports a case. */
  results?: LiveFeedResult[];
  /** Alarms a stop interrupted between people. The cursor did NOT advance for
   * these, so they are judged again, whole, on the next run. */
  partial_frames?: { clip_key: string; run_id: string; calls: number; unjudged_people: number }[];
  error?: string;
  poll_seconds?: number;
}

export interface LiveFeedResult {
  clip_key: string;
  alarm_id: string;
  frame_key: string | null;
  /** The audit case this alarm became — the image. `null` on a result
   * recorded before this field existed. */
  case_id?: string | null;
  /** The FRAME's verdict, OR'd over its people: any one in violation makes
   * the frame a violation. `null` when nothing in it was decided. */
  verdict: string | null;
  /** Boxes recovered from the vendor's snapshot, and how many were judged.
   * The two differ when a call failed. */
  people: number;
  judged: number;
  gated: number;
  failed: number;
  /** Every paid call for this alarm. */
  calls: number;
  /** When THIS alarm finished being judged. Stamped per alarm: it used to
   * carry the poll's own start time, so a whole batch shared one value. */
  at: string;
  /** When the alarm was RAISED, off the snapshot key. Absent on results
   * recorded before it was carried. */
  alarm_at?: string | null;
}

/** Whether the live feed can run on THIS deployment at all.
 *
 * On the app server the answer is no, by design: production footage stays off
 * it, so the instance has no credentials for the clip bucket. The feed is run
 * from a machine that does — the same place the extraction pass runs. Asked
 * before the Start button is offered, because a control that can only fail is
 * worse than one that explains why it is absent. */
export interface LiveFeedCapability {
  available: boolean;
  bucket: string;
  /** Empty when available; otherwise the whole explanation, in the server's
   * own words. */
  detail: string;
}

export function getLiveFeedCapability(): Promise<LiveFeedCapability> {
  return apiGet<LiveFeedCapability>("/track1/live-feed/capability");
}

export function getLiveFeedStatus(): Promise<LiveFeedState> {
  return apiGet<LiveFeedState>("/track1/live-feed/status");
}

export function startLiveFeed(params: {
  violation_type: "helmet";
  per_run_call_cap: number;
  daily_call_cap: number;
  /** `now` is the operator's feed: alarms arriving from this moment.
   * `oldest` resumes from the watermark, which on a first run is the start
   * of the corpus -- the only way to exercise this against alarms that
   * already exist. `date` takes everything on or after `start_date`. */
  start_from: "now" | "oldest" | "date";
  start_date?: string | null;
  /** Which examples the judge is shown first. `fixed` is the committed
   * library; `sampled` builds a balanced set from cases audited here, topping
   * a short side up from the committed one. The committed library is never
   * edited either way. */
  few_shot_source: "fixed" | "sampled";
  /** Where each person's box comes from. `vendor_cv` reads the rectangles the
   * vendor's own CV drew on its snapshot; `detector` runs our own person
   * model over the clean frame. Either way the SAME alarms are judged, so the
   * two are comparable, and every row records which produced it. */
  box_source: "vendor_cv" | "detector";
}): Promise<LiveFeedState> {
  return apiPost<LiveFeedState>("/track1/live-feed/start", params);
}

/** Clears the failure records, the skip list and the attempt counts. The
 * WATERMARK and the processed-alarm record are untouched — those are what stop
 * an alarm being judged and paid for twice. Stepped-over clips become eligible
 * again, which is the point of clearing. */
export function clearLiveFeedFailures(): Promise<LiveFeedState> {
  return apiPost<LiveFeedState>("/track1/live-feed/clear-failures");
}

export function stopLiveFeed(): Promise<LiveFeedState> {
  return apiPost<LiveFeedState>("/track1/live-feed/stop");
}

export function helmetCvCandidateImagePath(
  cvRunId: string,
  frameName: string,
  candidateId: string,
): string {
  return `/helmet-cv/runs/${encodeURIComponent(cvRunId)}/frames/${encodeURIComponent(frameName)}/candidates/${encodeURIComponent(candidateId)}/image`;
}

/** Records the reviewer's own verdict on one frame, and only then does the
 * server disclose CV's state and the judge's verdict for it. The response
 * carries the frame back with all three calls filled in; the screen refetches
 * the alarm as well, so the ruler's other ticks stay in step. */
export function reviewHelmetCvFrame(
  frameName: string,
  cvRunId: string,
  verdict: string,
  reviewer: string,
  note: string,
  candidateId: string | null,
): Promise<HelmetCvFrameRow> {
  return apiPost<HelmetCvFrameRow>(
    `/helmet-cv/frames/${encodeURIComponent(frameName)}/review?${new URLSearchParams({ cv_run_id: cvRunId })}`,
    { verdict, reviewer: reviewer || null, note: note || null, candidate_id: candidateId },
  );
}

/** `cvRunId` empty means EVERY run, which is what the rolling-metrics view
 * asks for. Each live run gets its own id, so filtering to one scores a
 * handful of audited frames and reports a precision off three cases. */
export function getHelmetCvSummary(
  cvRunId: string,
  compareRunId?: string | null,
): Promise<HelmetCvSummary> {
  const q = new URLSearchParams();
  if (cvRunId) q.set("cv_run_id", cvRunId);
  if (compareRunId) q.set("compare_run_id", compareRunId);
  return apiGet<HelmetCvSummary>(`/helmet-cv/summary?${q}`);
}

/** CV's state as the reader meets it: a word, in the mono instrument
 * treatment. Upper case because this is a readout off an instrument, not
 * prose -- and it is the one thing keeping WHITE / YELLOW / RED legible
 * without a swatch. An unrecognised state still renders, as its raw value. */
const CV_STATE_WORD: Record<string, string> = {
  white: "WHITE",
  yellow: "YELLOW",
  red: "RED",
  unknown: "UNKNOWN",
  no_helmet: "NO HELMET",
};

export function helmetCvStateWord(state: string): string {
  return CV_STATE_WORD[state] ?? state.replace(/_/g, " ").toUpperCase();
}

/** The server's own scoring word for one call against the reviewer's verdict.
 * Mapped where the wording is known and humanised otherwise, rather than
 * guessed at: this figure is the server's, and inventing a synonym for a
 * value it adds later would be this screen claiming to know what it means. */
const HELMET_CV_OUTCOME_LABEL: Record<string, string> = {
  correct: "Matched your call",
  incorrect: "Missed your call",
  match: "Matched your call",
  mismatch: "Missed your call",
  not_scored: "Not scored",
};

export function helmetCvOutcomeLabel(outcome: string): string {
  return HELMET_CV_OUTCOME_LABEL[outcome] ?? outcome.replace(/_/g, " ");
}

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
}

export interface Track2QueueEntry {
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
  pending_message: string | null;
  gpu_jobs: Track2GpuJob[];
  queue: Track2QueueSnapshot;
  dataset_browser: { prefix: string; folders: string[] };
  weights_browser: { prefix: string; folders: string[] };
  eval_browser: { prefix: string; folders: string[] };
}

/** One audit row per submitted run. Written at submit time with what the
 * operator chose, then completed by the server once the run finishes --
 * so `ended_at`, `model` and `metrics` are null until then. */
export interface Track2TrainingAuditDataset {
  folder: string;
  s3_uri: string;
  file_count: number;
  file_names: string[];
  /** file_names is a capped sample, not the whole set -- show file_count. */
  truncated: boolean;
}

export interface Track2TrainingAudit {
  id: string;
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
  train_dataset: Track2TrainingAuditDataset;
  eval_dataset: Track2TrainingAuditDataset;
  /** True when no separate evaluation set was chosen, so the run was
   * evaluated on its own training data. */
  eval_is_train_dataset: boolean;
  base_model: { folder: string; source: string };
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
}

export function startTrack2Training(params: Track2TrainingParams): Promise<{ message: string; build_id: string }> {
  return apiPost("/track2/models/train/dataset", params);
}

export function stopTrack2Training(): Promise<{ message: string }> {
  return apiPost("/track2/models/train/stop");
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
  size: string;
  version: string;
  model_package_arn: string;
  status: string;
  metrics: {
    mAP50: number | null;
    "mAP50-95": number | null;
    precision: number | null;
    recall: number | null;
    accuracy: number | null;
  } | null;
  description: string;
  lineage: Track2Lineage | null;
}

export function getTrack2Metrics(): Promise<{ rows: Track2MetricsRow[] }> {
  return apiGet("/track2/metrics");
}

/** The 8-color CVD-safe palette the old app used for the metrics
 * comparison chart -- kept exact for continuity with historical charts. */
export const TRACK2_COMPARE_COLORS = [
  "#3987e5", "#d95926", "#199e70", "#c98500",
  "#d55181", "#008300", "#9085e9", "#e66767",
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
}

export function getTrack2Notifications(): Promise<{ subscriptions: Track2NotificationSubscription[] }> {
  return apiGet("/track2/notifications");
}

export function addTrack2Notification(email: string): Promise<{ email: string; message: string }> {
  return apiPost("/track2/notifications/add", { email });
}

export function removeTrack2Notification(subscriptionArn: string): Promise<{ removed: boolean }> {
  return apiPost("/track2/notifications/remove", { subscription_arn: subscriptionArn });
}

export interface Track2ExportResponse {
  registries: Track2RegistriesBrief;
  folders: string[];
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
