import {
  BarChart3,
  CalendarDays,
  Database,
  DatabaseZap,
  Layers,
  Bell,
  Factory,
  HelpCircle,
  LogOut,
  Menu,
  Rocket,
  X,
  type LucideIcon,
} from "lucide-react";
import { useEffect, useRef, useState, type MouseEvent } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { getTrack2Build, type Track2Build } from "../lib/api";
import { useAuth } from "../context/AuthContext";
import { useNavGuard } from "../context/NavGuardContext";
import { RouteBoundary } from "./RouteBoundary";

interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  /** How finished the tab is, shown as the entry's colour. None: neutral. */
  status?: NavStatus;
}

type NavStatus = "ready" | "prototype" | "kiv";

const STATUS_STYLE: Record<NavStatus, { className: string; title: string }> = {
  ready: { className: "text-status-ready", title: "In use" },
  prototype: { className: "text-status-prototype", title: "Prototype -- still changing" },
  kiv: { className: "text-status-kiv", title: "Kept in view -- not being worked on" },
};

// The domains, in a fixed order -- this is confirmed IA, not something to
// redesign. Audit sits directly after Evaluate escalations because it
// reviews what those queues settled, and Training set sits between Run
// pipeline and Evaluate escalations because it is what a run's settled
// cases publish into -- but every entry is styled flush with the others,
// never indented: an indent implies a hierarchy the nav does not have (the
// list is flat, there is no nested routing anywhere in this app) and made
// the nested-looking entry read as misaligned rather than as belonging.
const NAV_ITEMS: NavItem[] = [
  // Track2 (YOLO training pipeline) only. The original PPE judge pages --
  // Run pipeline, Training set, escalations, Audit, Rolling metrics, Eval,
  // Few-shot, Gold Set, Model deployment status -- were removed from the
  // nav and routes; their page files remain in the tree but are no longer
  // reachable.
  // Track2 Training dropped from the nav. It and Multi-Auto Training are
  // the SAME page -- one component, `variant="multi"` the only
  // difference -- so two entries were two doors into one room, and the
  // work happens on the multi side. Route kept; `/` now lands on the
  // entry that is offered rather than on one that is not.
  // The data side, ahead of training because it comes first: what a run
  // learns from is prepared here, then trained on next door (2026-10-28).
  { to: "/track2/data-preparation", label: "Track2 Data Preparation", icon: Database, status: "ready" },
  { to: "/track2/multi-auto-training", label: "Track2 Multi-Auto Training", icon: Layers, status: "ready" },
  // Explore a Postgres database for images (2026-10-28). No status colour
  // yet -- not decided.
  { to: "/track2/sql", label: "Track2 SQL Database Query", icon: DatabaseZap },
  // What the clocks will start, and when -- read-only, every control
  // stays on Multi-Auto Training.
  { to: "/track2/scheduled", label: "Track2 Scheduled - prototype", icon: CalendarDays, status: "prototype" },
  // Schedule, Export and Cost dropped from the nav. Schedule in
  // particular was a SECOND retrain schedule: it drove the full-frame
  // replay coordinator on production lineage while the pull toolbar
  // drives the open pipeline on a detector's taxonomy, and neither page
  // mentioned the other. Setting both gave two unrelated retrains, with
  // the single-GPU guard refusing whichever lost -- naming a run nobody
  // started. Multi-Auto Training's toolbar is now the only schedule.
  //
  // Routes kept, which is the pattern the retired PPE judge paths use: a
  // bookmark or a link in somebody's notes still resolves instead of
  // 404ing, it just is not offered any more.
  // Lineage dropped from the nav, same as Schedule, Export and Cost
  // above: the route is kept, so a bookmark or a link in somebody's
  // notes still resolves rather than 404ing -- it just is not offered.
  { to: "/track2/metrics", label: "Track2 Review & Metrics", icon: BarChart3, status: "prototype" },
  // Evaluation dropped from the nav. Route kept, same as the others.
  { to: "/track2/notifications", label: "Track2 Notifications", icon: Bell },
  { to: "/track2/help", label: "Track2 Help", icon: HelpCircle },
  // Synthetic Image Generation dropped from the nav. Route kept, same as
  // the others.
  // Last, and marked KIV (kept in view): deploying and trying a model is
  // not part of the retraining workflow the entries above walk through.
  { to: "/track2/deploy", label: "Track2 Deploy & Inference (KIV)", icon: Rocket, status: "kiv" },
];

function Wordmark() {
  return (
    <div className="flex items-center gap-2">
      <Factory size={20} className="text-accent" strokeWidth={2} />
      <div className="leading-tight">
        <p className="text-sm font-semibold text-ink">Training and Development</p>
        <p className="tabular text-xs text-ink-muted">Jurong Port &middot; case review</p>
      </div>
    </div>
  );
}

function NavList({ onNavigate }: { onNavigate?: () => void }) {
  const { logout } = useAuth();
  const { isDirty } = useNavGuard();

  // Sidebar nav lives outside the routed page, so a page's own in-page
  // dirty-check (GoldsetPage's `guarded()`/`pendingNav`, today the only
  // registrant) previously had no way to intercept a sidebar click -- a
  // reviewer with an unsaved annotation draft could lose it silently by
  // clicking any other tab. This asks the currently-registered guard (if
  // any) before letting react-router actually swap the route out.
  function handleNavClick(e: MouseEvent<HTMLAnchorElement>) {
    if (isDirty() && !window.confirm("You have an unsaved annotation. Leave anyway?")) {
      e.preventDefault();
      return;
    }
    onNavigate?.();
  }

  return (
    <>
      <nav className="flex flex-1 flex-col gap-0.5 p-3">
        {NAV_ITEMS.map(({ to, label, icon: Icon, status }) => (
          <NavLink
            key={to}
            to={to}
            title={status ? STATUS_STYLE[status].title : undefined}
            // `end` so /escalations does not stay highlighted while its own
            // sub-entry is the active route -- react-router matches prefixes
            // by default, which would light up both.
            end
            onClick={handleNavClick}
            className={({ isActive }) =>
              `flex items-center gap-2.5 rounded-sm px-3 py-2 text-sm transition-colors ${
                isActive ? "bg-accent-subtle font-medium" : "hover:bg-black/[0.04]"
              } ${
                // The status colour REPLACES the neutral ink rather than
                // sitting beside it: two text colours on one element are
                // settled by the stylesheet's order, not this string's.
                status ? STATUS_STYLE[status].className : isActive ? "text-ink" : "text-ink-secondary"
              }`
            }
          >
            <Icon size={16} strokeWidth={2} />
            {label}
          </NavLink>
        ))}
      </nav>

      <button
        onClick={logout}
        className="flex items-center gap-2.5 border-t border-hairline px-5 py-3.5 text-sm text-ink-secondary hover:bg-black/[0.04]"
      >
        <LogOut size={16} strokeWidth={2} />
        Log out
      </button>
    </>
  );
}

/**
 * Persistent sidebar on md+ viewports (the case-file ledger doesn't get a
 * hamburger on desktop); below that, a slim top bar with a menu toggle
 * that opens the same nav as a full-height drawer -- same domains, same
 * order, no separate mobile IA.
 */
/**
 * Which build this is, and whether it is the stable container or a
 * development one.
 *
 * Two containers may run side by side on this host, on different ports,
 * against the SAME tables and the same bucket. That is only safe while
 * the difference is visible: the failure it guards against is a stale
 * build serving the URL people actually have, which nobody can see is
 * stale. A development container is coloured, a stable one is not, so a
 * glance is enough.
 *
 * Silent when the API does not answer -- an older image has no /build,
 * and a missing badge must not look like a broken page.
 */
/** How often to ask the server which build it is serving.
 *
 * A page left open keeps running the JavaScript it loaded, whatever has
 * been deployed since -- `index.html` is `no-cache`, but that only
 * matters on a reload. A tab open across a deploy therefore calls the
 * new API with the old code, and the mismatch is invisible: it looks
 * like a bug in the feature rather than a stale page. That cost a real
 * debugging round trip, hence this.
 *
 * Two minutes: deploys here take about seven, so this notices within the
 * next one and costs one tiny unauthenticated request.
 */
const BUILD_POLL_MS = 30000;

/** Shared so the sidebar badge and the banner across the top cannot
 * disagree about whether this page is stale. */
function useBuildFreshness() {
  const [build, setBuild] = useState<Track2Build | null>(null);
  // The tag this PAGE was loaded against, pinned on the first answer.
  const loadedTag = useRef("");
  const [serving, setServing] = useState("");

  useEffect(() => {
    let cancelled = false;
    const check = () => {
      getTrack2Build()
        .then((got) => {
          if (cancelled || !got.tag) return;
          if (!loadedTag.current) {
            loadedTag.current = got.tag;
            setBuild(got);
          }
          setServing(got.tag);
        })
        .catch(() => undefined);
    };
    check();
    const timer = window.setInterval(check, BUILD_POLL_MS);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, []);

  return { build, serving, loadedTag: loadedTag.current,
           stale: serving !== "" && loadedTag.current !== "" && serving !== loadedTag.current };
}

/** A band across the top of the content when the page is out of date.
 *
 * The sidebar already carried a small "New build -- reload" button, and
 * it was missed across four separate rounds of "this change has not
 * appeared": each time the fix was deployed and verified in the running
 * container, and each time the tab had been open since before the
 * deploy, running the old JavaScript against the new API.
 *
 * A control that is routinely missed is the wrong control, however
 * correct. This sits where the content is, in the colour reserved for
 * things that are wrong, and says what it will do.
 */
export function StaleBuildBanner() {
  const { serving, loadedTag, stale } = useBuildFreshness();
  if (!stale) return null;
  return (
    <div className="sticky top-0 z-40 flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-critical/30 bg-critical-subtle px-4 py-2 text-xs text-critical md:px-8">
      <span className="font-semibold">This page is out of date.</span>
      <span className="tabular text-critical/80">
        running {loadedTag} &middot; the server is serving {serving}
      </span>
      <button
        type="button"
        onClick={() => window.location.reload()}
        className="ml-auto rounded-md bg-critical px-3 py-1 text-xs font-medium text-white"
      >
        Reload
      </button>
    </div>
  );
}

function BuildBadge() {
  const { build, serving, loadedTag, stale } = useBuildFreshness();

  if (!build) return null;
  const development = build.role !== "" && build.role !== "stable";
  return (
    <>
      <p
        className={`tabular mt-1 truncate text-[11px] ${
          development ? "font-semibold text-critical" : "text-ink-muted"
        }`}
        title={`image ${build.tag}${build.port ? ` on port ${build.port}` : ""}`}
      >
        {development && "\u25CF "}
        {build.role || "build"} · {build.tag}
      </p>
      {stale && (
        // Not reloaded automatically: a reload throws away whatever is
        // half-typed in a card, and losing a folder list nobody had
        // saved yet is worse than running a build one deploy behind.
        <button
          type="button"
          onClick={() => window.location.reload()}
          className="mt-1 w-full rounded-md border border-critical/40 bg-critical-subtle px-2 py-1 text-left text-[11px] font-medium text-critical"
          title={`this page is running ${loadedTag}; the server is serving ${serving}`}
        >
          New build {serving} — reload
        </button>
      )}
    </>
  );
}

export function Layout() {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const location = useLocation();

  return (
    /* `md:h-svh` -- a FIXED height from the medium breakpoint up, not
       just a minimum. With `min-h-svh` the shell grows with its content
       and the DOCUMENT scrolls, so <main>'s own `overflow-y-auto` never
       engages and anything `sticky` inside it sticks to a container that
       is itself scrolling away. Pinning the toolbar needs main to be the
       real scroll container.
       Mobile keeps `min-h-svh`: there the nav stacks above the content
       and a fixed-height shell would squash it. */
    <div className="flex min-h-svh flex-col md:h-svh md:flex-row">
      <header className="flex items-center justify-between border-b border-hairline bg-surface px-4 py-3 md:hidden">
        <div className="min-w-0">
          <Wordmark />
          <BuildBadge />
        </div>
        <button
          onClick={() => setDrawerOpen(true)}
          aria-label="Open navigation"
          className="rounded-sm border border-hairline p-3 text-ink-secondary"
        >
          <Menu size={20} />
        </button>
      </header>

      <aside className="hidden w-64 shrink-0 flex-col border-r border-hairline bg-surface md:flex">
        <div className="border-b border-hairline px-5 py-4">
          <Wordmark />
          <BuildBadge />
        </div>
        <NavList />
      </aside>

      {drawerOpen && (
        <div className="fixed inset-0 z-20 flex md:hidden">
          <div className="absolute inset-0 bg-ink/40" onClick={() => setDrawerOpen(false)} />
          <aside className="relative flex w-72 max-w-[85vw] flex-col bg-surface">
            <div className="flex items-center justify-between border-b border-hairline px-5 py-4">
              <Wordmark />
              <button
                onClick={() => setDrawerOpen(false)}
                aria-label="Close navigation"
                className="rounded-sm p-3 text-ink-secondary"
              >
                <X size={20} />
              </button>
            </div>
            <NavList onNavigate={() => setDrawerOpen(false)} />
          </aside>
        </div>
      )}

      <main className="flex-1 overflow-y-auto bg-page">
        <StaleBuildBanner />

        <div className="mx-auto max-w-6xl px-4 py-6 md:px-8 md:py-8">
          {/* Keyed by path so navigating away clears a failure: a latched
              boundary would make every page after a broken one look
              broken too. */}
          <RouteBoundary key={location.pathname}>
            <Outlet />
          </RouteBoundary>
        </div>
      </main>
    </div>
  );
}
