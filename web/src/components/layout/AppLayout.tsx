import { useQuery } from "@tanstack/react-query";
import { Activity, BookOpen, FlaskConical, Menu, MessageSquare, Upload } from "lucide-react";
import { Dialog as D } from "radix-ui";
import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router";
import { api } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Banner } from "@/components/ui/states";
import { formatUsd } from "@/lib/format";
import { JobsProvider } from "@/lib/jobs";
import { cn } from "@/lib/utils";
import { AccountMenu } from "./AccountMenu";
import { Logo } from "./Logo";

const NAV = [
  { to: "/", label: "Chat", icon: MessageSquare, end: true, match: (p: string) => p === "/" || p.startsWith("/c/") },
  { to: "/library", label: "Library", icon: BookOpen },
  { to: "/upload", label: "Upload", icon: Upload },
  { to: "/evaluation", label: "Evaluation", icon: FlaskConical },
  { to: "/traces", label: "Traces", icon: Activity },
];

function NavItems({ onNavigate }: { onNavigate?: () => void }) {
  const { pathname } = useLocation();
  return (
    <nav aria-label="Main" className="flex flex-col gap-0.5">
      {NAV.map(({ to, label, icon: Icon, end, match }) => (
        <NavLink
          key={to}
          to={to}
          end={end}
          onClick={onNavigate}
          className={({ isActive }) =>
            cn(
              "flex h-9 items-center gap-2.5 rounded-lg px-2.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-subtle hover:text-foreground",
              (match ? match(pathname) : isActive) && "bg-subtle text-foreground",
            )
          }
        >
          <Icon className="size-4" aria-hidden />
          {label}
        </NavLink>
      ))}
    </nav>
  );
}

function Budget() {
  const health = useQuery({ queryKey: ["health"], queryFn: () => api.get("/health"), refetchInterval: 60_000 });
  const h = health.data;
  if (!h || h.month_cap_usd === null || h.month_cap_usd === undefined) return null;
  const spent = h.month_spend_usd ?? 0;
  const frac = Math.min(1, spent / (h.month_cap_usd || 1));
  return (
    <div className="rounded-lg border border-border bg-surface px-3 py-2.5 text-xs">
      <div className="flex items-baseline justify-between text-muted-foreground">
        <span>This month</span>
        <span className="tabular text-foreground">
          {formatUsd(spent)} <span className="text-muted-foreground">/ {formatUsd(h.month_cap_usd)}</span>
        </span>
      </div>
      <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-subtle" aria-hidden>
        <div className={cn("h-full rounded-full", frac >= 1 ? "bg-danger" : frac > 0.8 ? "bg-warning" : "bg-accent")} style={{ width: `${frac * 100}%` }} />
      </div>
      {h.active_profile ? <p className="mt-1.5 truncate text-muted-foreground">Profile {h.active_profile}</p> : null}
    </div>
  );
}

function RetrievalOnlyBanner() {
  const health = useQuery({ queryKey: ["health"], queryFn: () => api.get("/health"), refetchInterval: 60_000 });
  if (!health.data?.retrieval_only) return null;
  return (
    <Banner testId="banner-retrieval-only" title="Retrieval-only mode" className="mx-4 mt-3 sm:mx-6">
      This month's knowledge budget is used up, so questions get the closest passages instead of a written answer. Search still works.
    </Banner>
  );
}

export function AppLayout() {
  const [menuOpen, setMenuOpen] = useState(false);
  const { pathname } = useLocation();
  useEffect(() => setMenuOpen(false), [pathname]);
  const current = NAV.find((n) => (n.match ? n.match(pathname) : pathname.startsWith(n.to)))?.label ?? "HAL";

  return (
    <JobsProvider>
      <div className="flex min-h-dvh">
        <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:rounded-md focus:bg-surface focus:px-3 focus:py-2 focus:shadow-pop">
          Skip to content
        </a>
        {/* Desktop sidebar */}
        <aside className="sticky top-0 hidden h-dvh w-60 shrink-0 flex-col border-r border-border bg-surface-2/60 px-3 py-4 lg:flex">
          <Logo className="px-2" />
          <div className="mt-6 flex-1">
            <NavItems />
          </div>
          <div className="space-y-2">
            <Budget />
            <AccountMenu className="w-full" />
          </div>
        </aside>

        <div className="flex min-w-0 flex-1 flex-col">
          {/* Mobile top bar */}
          <header className="sticky top-0 z-30 flex h-14 items-center gap-2 border-b border-border bg-background/85 px-3 backdrop-blur lg:hidden">
            <D.Root open={menuOpen} onOpenChange={setMenuOpen}>
              <D.Trigger asChild>
                <Button variant="ghost" size="icon" aria-label="Open menu">
                  <Menu />
                </Button>
              </D.Trigger>
              <D.Portal>
                <D.Overlay className="fixed inset-0 z-40 bg-black/40" />
                <D.Content className="fixed inset-y-0 left-0 z-50 flex w-72 max-w-[85vw] flex-col bg-surface px-3 py-4 shadow-pop outline-none">
                  <D.Title className="sr-only">Menu</D.Title>
                  <D.Description className="sr-only">Pages</D.Description>
                  <Logo className="px-2" />
                  <div className="mt-6 flex-1">
                    <NavItems onNavigate={() => setMenuOpen(false)} />
                  </div>
                  <div className="space-y-2">
                    <Budget />
                    <AccountMenu className="w-full" />
                  </div>
                </D.Content>
              </D.Portal>
            </D.Root>
            <span className="text-[15px] font-semibold">{current}</span>
          </header>

          <RetrievalOnlyBanner />
          <main id="main" className="flex min-h-0 min-w-0 flex-1 flex-col">
            <Outlet />
          </main>
        </div>
      </div>
    </JobsProvider>
  );
}
