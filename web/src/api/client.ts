/**
 * A thin typed client over the generated OpenAPI types (schema.d.ts).
 *
 * Every response type is derived from `paths`, so a change to openapi.json
 * that the UI hasn't caught up with fails `npm run typecheck`.
 */
import type { components, paths } from "./schema";

export type Schemas = components["schemas"];

type Method = "get" | "post" | "put" | "patch" | "delete";

type Json200<Op> = Op extends { responses: { 200: { content: { "application/json": infer R } } } } ? R : void;

/** The JSON body of a route's 200 response (void for 204-only routes). */
export type ResponseOf<P extends keyof paths, M extends Method> = paths[P] extends Record<M, infer Op> ? Json200<Op> : never;

/** The JSON request body of a route. */
export type BodyOf<P extends keyof paths, M extends Method> =
  paths[P] extends Record<M, infer Op>
    ? Op extends { requestBody: { content: { "application/json": infer B } } }
      ? B
      : never
    : never;

/** Query parameters of a route. */
export type QueryOf<P extends keyof paths, M extends Method> =
  paths[P] extends Record<M, infer Op> ? (Op extends { parameters: { query?: infer Q } } ? Q : never) : never;

export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;
  constructor(status: number, message: string, detail?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

type UnauthorizedListener = () => void;
const unauthorizedListeners = new Set<UnauthorizedListener>();

/** Called whenever the server answers 401 (the session expired or never existed). */
export function onUnauthorized(listener: UnauthorizedListener): () => void {
  unauthorizedListeners.add(listener);
  return () => unauthorizedListeners.delete(listener);
}

export function notifyUnauthorized(): void {
  for (const l of unauthorizedListeners) l();
}

export function buildUrl(path: string, params?: Record<string, string | number>, query?: object): string {
  let url = path.replace(/\{(\w+)\}/g, (_m, key: string) => {
    const v = params?.[key];
    if (v === undefined) throw new Error(`missing path parameter ${key} for ${path}`);
    return encodeURIComponent(String(v));
  });
  if (query) {
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(query)) {
      if (v === undefined || v === null || v === "") continue;
      qs.set(k, String(v));
    }
    const s = qs.toString();
    if (s) url += `?${s}`;
  }
  return url;
}

export function errorMessage(status: number, body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const d = (body as { detail: unknown }).detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d) && d.length && typeof d[0] === "object" && d[0] && "msg" in d[0]) {
      return String((d[0] as { msg: unknown }).msg);
    }
  }
  if (status === 401) return "Please sign in again.";
  if (status === 501) return "This part of the server isn't available yet.";
  if (status >= 500) return "The server had a problem. Try again in a moment.";
  return fallback;
}

export async function readError(res: Response): Promise<ApiError> {
  let body: unknown = undefined;
  try {
    body = await res.json();
  } catch {
    /* not JSON */
  }
  return new ApiError(res.status, errorMessage(res.status, body, res.statusText || `HTTP ${res.status}`), body);
}

export interface RequestOptions<P extends keyof paths, M extends Method> {
  params?: Record<string, string | number>;
  query?: QueryOf<P, M>;
  body?: BodyOf<P, M>;
  signal?: AbortSignal;
}

export async function request<P extends keyof paths, M extends Method>(
  method: M,
  path: P,
  opts: RequestOptions<P, M> = {},
): Promise<ResponseOf<P, M>> {
  const url = buildUrl(path as string, opts.params, opts.query as object | undefined);
  let res: Response;
  try {
    res = await fetch(url, {
      method: method.toUpperCase(),
      credentials: "same-origin",
      headers: opts.body !== undefined ? { "Content-Type": "application/json", Accept: "application/json" } : { Accept: "application/json" },
      body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
      signal: opts.signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError(0, "Can't reach the house server. Check that it's running and you're on the home network.");
  }
  if (!res.ok) {
    const err = await readError(res);
    if (res.status === 401 && path !== "/api/session") notifyUnauthorized();
    throw err;
  }
  if (res.status === 204) return undefined as ResponseOf<P, M>;
  const text = await res.text();
  return (text ? JSON.parse(text) : undefined) as ResponseOf<P, M>;
}

export const api = {
  get: <P extends keyof paths>(path: P, opts?: RequestOptions<P, "get">) => request("get", path, opts),
  post: <P extends keyof paths>(path: P, opts?: RequestOptions<P, "post">) => request("post", path, opts),
  put: <P extends keyof paths>(path: P, opts?: RequestOptions<P, "put">) => request("put", path, opts),
  patch: <P extends keyof paths>(path: P, opts?: RequestOptions<P, "patch">) => request("patch", path, opts),
  delete: <P extends keyof paths>(path: P, opts?: RequestOptions<P, "delete">) => request("delete", path, opts),
};

/** URL of an original file (PDFs are served inline with range support). */
export const fileUrl = (sha: string) => buildUrl("/api/files/{sha}", { sha });
