/**
 * Upload with per-file progress and a concurrency limit (three at a time,
 * plan §5.2). fetch() has no upload progress, so each file goes up with
 * XMLHttpRequest as one multipart POST /api/uploads (files + relative_path).
 */
import { ApiError, errorMessage, notifyUnauthorized } from "@/api/client";
import type { UploadResult } from "@/api/types";

export const UPLOAD_CONCURRENCY = 3;

export function uploadOne(
  file: File,
  relativePath: string,
  onProgress: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<UploadResult> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/uploads");
    xhr.withCredentials = true;
    xhr.responseType = "json";
    xhr.setRequestHeader("Accept", "application/json");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      const body = xhr.response as { results?: UploadResult[] } | null;
      if (xhr.status >= 200 && xhr.status < 300 && body?.results?.length) {
        onProgress(1);
        resolve(body.results[0]);
      } else {
        if (xhr.status === 401) notifyUnauthorized();
        reject(new ApiError(xhr.status, errorMessage(xhr.status, body, `Upload failed (HTTP ${xhr.status})`), body));
      }
    };
    xhr.onerror = () => reject(new ApiError(0, "Network error while uploading."));
    xhr.onabort = () => reject(new DOMException("Upload cancelled", "AbortError"));
    signal?.addEventListener("abort", () => xhr.abort(), { once: true });
    const form = new FormData();
    form.append("files", file, file.name);
    form.append("relative_path", relativePath);
    xhr.send(form);
  });
}

/**
 * A concurrency limiter shared by everything that uploads: new batches and
 * retries queue behind what's already running, so there are never more than
 * `limit` uploads in flight in total.
 */
export class TaskQueue {
  private active = 0;
  private waiting: (() => void)[] = [];
  constructor(readonly limit: number) {}

  get inFlight(): number {
    return this.active;
  }

  run<T>(task: () => Promise<T>): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const start = () => {
        this.active++;
        task()
          .then(resolve, reject)
          .finally(() => {
            this.active--;
            this.waiting.shift()?.();
          });
      };
      if (this.active < this.limit) start();
      else this.waiting.push(start);
    });
  }
}

/** The app-wide upload queue (plan §5.2: three files at a time). */
export const uploadQueue = new TaskQueue(UPLOAD_CONCURRENCY);

/** Run `task` over `items` with at most `limit` in flight; resolves when all settle. */
export async function runPool<T>(items: T[], limit: number, task: (item: T, index: number) => Promise<void>): Promise<void> {
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const i = next++;
      await task(items[i], i);
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
}
