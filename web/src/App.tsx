import { QueryClientProvider } from "@tanstack/react-query";
import { lazy, Suspense, useEffect, useState } from "react";
import { BrowserRouter, Navigate, Route, Routes, useLocation, useNavigate } from "react-router";
import { onUnauthorized } from "@/api/client";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Spinner } from "@/components/ui/states";
import { AppLayout } from "@/components/layout/AppLayout";
import { makeQueryClient } from "@/lib/query";
import { ChatPage } from "@/pages/ChatPage";
import { LibraryPage } from "@/pages/LibraryPage";
import { DocumentPage } from "@/pages/DocumentPage";
import { SignInPage } from "@/pages/SignInPage";
import { TracesPage } from "@/pages/TracesPage";
import { TraceInspectorPage } from "@/pages/TraceInspectorPage";
import { UploadPage } from "@/pages/UploadPage";
import { NotFoundPage } from "@/pages/NotFoundPage";

// Recharts is the heaviest dependency; load it with the Evaluation page only.
const EvaluationPage = lazy(() => import("@/pages/EvaluationPage"));

/** Any 401 sends you to Sign in, then back to where you were. */
function AuthWatcher({ onSignedOut }: { onSignedOut: () => void }) {
  const navigate = useNavigate();
  const location = useLocation();
  useEffect(
    () =>
      onUnauthorized(() => {
        if (location.pathname === "/sign-in") return;
        onSignedOut();
        const next = encodeURIComponent(location.pathname + location.search);
        navigate(`/sign-in?next=${next}`, { replace: true });
      }),
    [navigate, location, onSignedOut],
  );
  return null;
}

export function AppRoutes({ onSignedOut = () => {} }: { onSignedOut?: () => void }) {
  return (
    <>
      <AuthWatcher onSignedOut={onSignedOut} />
      <Routes>
        <Route path="/sign-in" element={<SignInPage />} />
        <Route element={<AppLayout />}>
          <Route index element={<ChatPage />} />
          <Route path="c/:conversationId" element={<ChatPage />} />
          <Route path="library" element={<LibraryPage />} />
          <Route path="library/:id" element={<DocumentPage />} />
          <Route path="upload" element={<UploadPage />} />
          <Route
            path="evaluation"
            element={
              <Suspense fallback={<Spinner className="p-8" />}>
                <EvaluationPage />
              </Suspense>
            }
          />
          <Route path="traces" element={<TracesPage />} />
          <Route path="traces/:id" element={<TraceInspectorPage />} />
          <Route path="chat" element={<Navigate to="/" replace />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </>
  );
}

export function App() {
  const [client] = useState(makeQueryClient);
  return (
    <QueryClientProvider client={client}>
      <TooltipProvider>
        <BrowserRouter>
          <AppRoutes onSignedOut={() => client.clear()} />
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
}
