import {
  BarChart3,
  Bell,
  CalendarClock,
  Cpu,
  Download,
  Factory,
  FlaskConical,
  GitBranch,
  HelpCircle,
  LogOut,
  Menu,
  Rocket,
  Sparkles,
  Wallet,
  X,
  type LucideIcon,
} from "lucide-react";
import { useState, type MouseEvent } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { useNavGuard } from "../context/NavGuardContext";

interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
}

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
  { to: "/track2/models", label: "Track2 Training", icon: Cpu },
  { to: "/track2/schedule", label: "Track2 Schedule", icon: CalendarClock },
  { to: "/track2/lineage", label: "Track2 Lineage", icon: GitBranch },
  { to: "/track2/metrics", label: "Track2 Review & Metrics", icon: BarChart3 },
  { to: "/track2/deploy", label: "Track2 Deploy & Inference", icon: Rocket },
  { to: "/track2/eval", label: "Track2 Evaluation", icon: FlaskConical },
  { to: "/track2/notifications", label: "Track2 Notifications", icon: Bell },
  { to: "/track2/export", label: "Track2 Export", icon: Download },
  { to: "/track2/cost", label: "Track2 Cost", icon: Wallet },
  { to: "/track2/help", label: "Track2 Help", icon: HelpCircle },
  { to: "/track2/synthetic", label: "Track2 Synthetic Image Generation", icon: Sparkles },
];

function Wordmark() {
  return (
    <div className="flex items-center gap-2">
      <Factory size={20} className="text-accent" strokeWidth={2} />
      <div className="leading-tight">
        <p className="text-sm font-semibold text-ink">PPE Safety Judge</p>
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
        {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            // `end` so /escalations does not stay highlighted while its own
            // sub-entry is the active route -- react-router matches prefixes
            // by default, which would light up both.
            end
            onClick={handleNavClick}
            className={({ isActive }) =>
              `flex items-center gap-2.5 rounded-sm px-3 py-2 text-sm transition-colors ${
                isActive
                  ? "bg-accent-subtle font-medium text-ink"
                  : "text-ink-secondary hover:bg-black/[0.04]"
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
export function Layout() {
  const [drawerOpen, setDrawerOpen] = useState(false);

  return (
    <div className="flex min-h-svh flex-col md:flex-row">
      <header className="flex items-center justify-between border-b border-hairline bg-surface px-4 py-3 md:hidden">
        <Wordmark />
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
        <div className="mx-auto max-w-6xl px-4 py-6 md:px-8 md:py-8">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
