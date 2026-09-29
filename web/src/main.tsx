import "./index.css";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";

async function boot() {
  // `npm run dev:mock` (and the Playwright mock runs) answer the API with MSW.
  if (import.meta.env.DEV && import.meta.env.VITE_MOCK === "1") {
    const { startMockWorker } = await import("./mocks/browser");
    await startMockWorker();
  }
  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}

void boot();
