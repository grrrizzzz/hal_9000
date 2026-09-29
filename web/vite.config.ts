/// <reference types="vitest/config" />
import { readFileSync } from "node:fs";
import { fileURLToPath, URL } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv, type Plugin } from "vite";

/**
 * Serves MSW's service worker at /mockServiceWorker.js in dev only, straight
 * from node_modules, so it is never committed and never lands in dist/.
 */
function mswWorker(): Plugin {
  return {
    name: "hal-msw-worker",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use("/mockServiceWorker.js", (_req, res) => {
        const path = fileURLToPath(new URL("./node_modules/msw/lib/mockServiceWorker.js", import.meta.url));
        res.setHeader("Content-Type", "text/javascript");
        res.setHeader("Service-Worker-Allowed", "/");
        res.end(readFileSync(path));
      });
    },
  };
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  // Plain `npm run dev` proxies the API to a running `hal9000 kb serve`
  // (KB_PORT defaults to 8910, plan §17). `npm run dev:mock` uses MSW instead.
  const backend = env.KB_DEV_BACKEND || `http://127.0.0.1:${env.KB_PORT || "8910"}`;
  return {
    plugins: [react(), tailwindcss(), mswWorker()],
    resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
    server: {
      proxy: mode === "mock" ? undefined : { "/api": backend, "/health": backend, "/metrics": backend },
    },
    build: {
      outDir: "dist",
      emptyOutDir: true,
      chunkSizeWarningLimit: 900,
      rollupOptions: {
        output: {
          manualChunks: {
            react: ["react", "react-dom", "react-router"],
            charts: ["recharts"],
            pdf: ["react-pdf"],
          },
        },
      },
    },
    test: {
      environment: "jsdom",
      globals: false,
      setupFiles: ["./src/test/setup.ts"],
      include: ["src/**/*.test.{ts,tsx}"],
      css: false,
      restoreMocks: true,
    },
  };
});
