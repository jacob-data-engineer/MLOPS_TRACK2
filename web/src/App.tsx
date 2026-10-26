/*
 * THESIS: Every reviewed case gets a real stamped verdict, not a color-coded
 * pill — this surface owns the customs-inspection ritual the product already
 * performs (dual review, adjudication, a final verdict), refusing the generic
 * clean-SaaS-dashboard default it already shipped once.
 * OWN-WORLD: Manila/carbon-paper neutrals as ground; one alarm-red ink stamp
 * reserved for verdict moments (VIOLATION / NO VIOLATION / ESCALATE stamped directly
 * onto the case, never a tinted pill); typewriter-monospace for case IDs,
 * coordinates, and counts; a plain workhorse sans for body copy and labels;
 * ledger-row tables with ruled hairlines; a hand-initialed adjudicator
 * sign-off field wherever a human decision lands.
 * STORY: A reviewer opens a case, sees prior reviewers' entries laid out like
 * carbon-copy ledger rows, adjudicates, and the verdict lands as a stamp —
 * auditable and legible at a glance, never a silent status change.
 * FIRST VIEWPORT: A case-file/ledger table fills the frame immediately on
 * every screen — no hero, no marketing chrome; a stamped verdict is visible
 * the instant one exists, not buried behind a click.
 * FORM: Customs case-file & manifest stamps — IMPECCABLE'S PICK (CEO's
 * explicit choice over the dice-assigned "container bay-plan grid"
 * direction), seed key 7ac63257.
 * FINISH: unreviewed and undocumented is unfinished; this build ends with
 * the finish review, the verdict, DESIGN.md, and every shipping raster
 * carrying its provenance.
 */
import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { RequireAuth } from "./components/RequireAuth";
import { RouteBoundary } from "./components/RouteBoundary";
import { LoginPage } from "./routes/LoginPage";
import { Track2CostPage } from "./routes/Track2CostPage";
import { Track2DeployPage } from "./routes/Track2DeployPage";
import { Track2EvalPage } from "./routes/Track2EvalPage";
import { Track2ExportPage } from "./routes/Track2ExportPage";
import { Track2HelpPage } from "./routes/Track2HelpPage";
import { Track2LineagePage } from "./routes/Track2LineagePage";
import { Track2MetricsPage } from "./routes/Track2MetricsPage";
import { Track2ModelsPage } from "./routes/Track2ModelsPage";
import { Track2NotificationsPage } from "./routes/Track2NotificationsPage";
import { Track2ScheduledPage } from "./routes/Track2ScheduledPage";
import { Track2SchedulePage } from "./routes/Track2SchedulePage";
import { Track2SyntheticPage } from "./routes/Track2SyntheticPage";

export function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />

      {/* The Help walkthrough, popped out. OUTSIDE the Layout route on
          purpose: a second window carrying the sidebar, the wordmark and
          the build badge is a second copy of the app, not a reference you
          can keep beside the page you are working in. Still behind
          RequireAuth -- it describes this estate, and the session is the
          same one. */}
      <Route
        path="/track2/help/popout"
        element={
          <RequireAuth>
            <div className="min-h-svh bg-page p-4">
              <RouteBoundary>
                <Track2HelpPage popout />
              </RouteBoundary>
            </div>
          </RequireAuth>
        }
      />

      <Route
        element={
          <RequireAuth>
            <Layout />
          </RequireAuth>
        }
      >
        {/* Track2 only. The original PPE judge pages (Run pipeline, Training
            set, escalations, Audit, Rolling metrics, Eval, Few-shot, Gold
            Set, Model deployment status) were removed from the product.
            Their old paths redirect to the landing page rather than 404,
            via the catch-all below, so existing links and bookmarks still
            land somewhere real. */}
        {/* The landing page has to be one the nav offers. `/track2/models`
            is still routed -- bookmarks resolve -- but it is no longer in
            the sidebar, so arriving there would show a page with nothing
            highlighted and no way back to it. */}
        <Route path="/" element={<Navigate to="/track2/multi-auto-training" replace />} />

        {/* Review folded into Metrics (approve/reject/delete now live next
            to the metrics table itself) -- an old link still lands
            somewhere real. */}
        <Route path="/track2/review" element={<Navigate to="/track2/metrics" replace />} />
        <Route path="/track2/models" element={<Track2ModelsPage />} />
        {/* The SAME component, so the queue, the GPU status and the
            history are shared rather than duplicated. */}
        <Route
          path="/track2/multi-auto-training"
          element={<Track2ModelsPage variant="multi" />}
        />
        {/* A suggested read-only view of the same settings as one picture.
            Nothing on it writes; every stage links back to its card. */}
        <Route path="/track2/schedule" element={<Track2SchedulePage />} />
        <Route path="/track2/lineage" element={<Track2LineagePage />} />
        <Route path="/track2/metrics" element={<Track2MetricsPage />} />
        <Route path="/track2/deploy" element={<Track2DeployPage />} />
        <Route path="/track2/eval" element={<Track2EvalPage />} />
        {/* Inference folded into Deploy (one endpoint-status banner, one
            place to both deploy and try it) -- an old link still lands
            somewhere real. */}
        <Route path="/track2/inference" element={<Navigate to="/track2/deploy" replace />} />
        <Route path="/track2/notifications" element={<Track2NotificationsPage />} />
        <Route path="/track2/scheduled" element={<Track2ScheduledPage />} />
        <Route path="/track2/help" element={<Track2HelpPage />} />
        <Route path="/track2/synthetic" element={<Track2SyntheticPage />} />
        <Route path="/track2/export" element={<Track2ExportPage />} />
        <Route path="/track2/cost" element={<Track2CostPage />} />
      </Route>

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
