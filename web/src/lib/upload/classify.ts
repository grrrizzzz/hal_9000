/**
 * The pre-upload screen (plan §5.2): what will be sent and what is skipped,
 * summarised as "37 files found: 30 will be added, 7 skipped (.jpg ×5, .heic ×2)".
 * The server screens again (magic bytes, size, duplicates); this only saves
 * uploading things that can't be indexed.
 */
import type { PickedFile } from "./walker";

export const ALLOWED_EXTENSIONS = ["pdf", "docx", "txt", "md", "html"] as const;
export const MAX_BYTES = 100 * 1024 * 1024;
const IMAGE_EXTENSIONS = new Set(["jpg", "jpeg", "png", "gif", "heic", "heif", "webp", "tif", "tiff", "bmp", "svg"]);

export type SkipReason = "image" | "unsupported" | "too_large" | "hidden" | "empty";

export interface Classified extends PickedFile {
  ext: string;
  skip: SkipReason | null;
}

export function extensionOf(name: string): string {
  const base = name.split("/").pop() ?? name;
  const dot = base.lastIndexOf(".");
  return dot > 0 ? base.slice(dot + 1).toLowerCase() : "";
}

export function classify(f: PickedFile): Classified {
  const ext = extensionOf(f.relativePath || f.file.name);
  const base = (f.relativePath.split("/").pop() ?? f.file.name) || f.file.name;
  let skip: SkipReason | null = null;
  if (base.startsWith(".") || f.relativePath.split("/").some((seg) => seg.startsWith(".") && seg.length > 1)) skip = "hidden";
  else if (IMAGE_EXTENSIONS.has(ext)) skip = "image";
  else if (!(ALLOWED_EXTENSIONS as readonly string[]).includes(ext)) skip = "unsupported";
  else if (f.file.size > MAX_BYTES) skip = "too_large";
  else if (f.file.size === 0) skip = "empty";
  return { ...f, ext, skip };
}

export const SKIP_LABEL: Record<SkipReason, string> = {
  image: "Image — text isn't read from pictures",
  unsupported: "Not a supported type",
  too_large: "Larger than 100 MB",
  hidden: "Hidden or system file",
  empty: "Empty file",
};

export interface Summary {
  total: number;
  added: Classified[];
  skipped: Classified[];
  /** Skipped files grouped by extension, most frequent first: [".jpg", 5]. */
  skippedByExt: [string, number][];
  hasImages: boolean;
  text: string;
}

export function summarize(files: PickedFile[]): Summary {
  const classified = dedupe(files).map(classify);
  const added = classified.filter((c) => !c.skip);
  const skipped = classified.filter((c) => c.skip);
  const counts = new Map<string, number>();
  for (const s of skipped) {
    const key = s.skip === "hidden" ? "hidden" : s.ext ? `.${s.ext}` : "no extension";
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  const skippedByExt = [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  const n = classified.length;
  let text = `${n} ${n === 1 ? "file" : "files"} found: ${added.length} will be added`;
  if (skipped.length) {
    text += `, ${skipped.length} skipped (${skippedByExt.map(([k, c]) => `${k} ×${c}`).join(", ")})`;
  }
  return { total: n, added, skipped, skippedByExt, hasImages: skipped.some((s) => s.skip === "image"), text };
}

/** The same path picked twice (e.g. dropped again) counts once. */
function dedupe(files: PickedFile[]): PickedFile[] {
  const seen = new Map<string, PickedFile>();
  for (const f of files) seen.set(`${f.relativePath}|${f.file.size}`, f);
  return [...seen.values()];
}
