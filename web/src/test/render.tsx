import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { AppRoutes } from "@/App";
import { TooltipProvider } from "@/components/ui/tooltip";
import { db } from "@/mocks/db";

/** Render the whole app at `route` against the MSW mocks, optionally signed in. */
export function renderApp(route: string, { signedIn = true } = {}) {
  db.signedIn = signedIn;
  db.tokenDelayMs = 2;
  db.jobTickMs = 20;
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <TooltipProvider>
        <MemoryRouter initialEntries={[route]}>
          <AppRoutes onSignedOut={() => client.clear()} />
        </MemoryRouter>
      </TooltipProvider>
    </QueryClientProvider>,
  );
}
