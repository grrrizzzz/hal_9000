/**
 * Walking dropped folders (plan §5.2).
 *
 * A drop's DataTransferItems expose `webkitGetAsEntry()`; directories are read
 * with `createReader().readEntries()`, which returns at most ~100 entries per
 * call and must be called until it returns an empty batch. Each file keeps its
 * path relative to what was dropped ("Appliances/Furnace/manual.pdf"), which
 * becomes the collection tag on the server.
 */

export interface PickedFile {
  file: File;
  /** Path relative to the dropped or chosen folder, "/"-separated, no leading slash. */
  relativePath: string;
}

// Minimal structural types, so tests can pass fakes (jsdom has no FileSystem API).
export interface FsEntry {
  isFile: boolean;
  isDirectory: boolean;
  name: string;
  fullPath: string;
}
export interface FsFileEntry extends FsEntry {
  file(success: (f: File) => void, error?: (e: unknown) => void): void;
}
export interface FsDirectoryEntry extends FsEntry {
  createReader(): { readEntries(success: (entries: FsEntry[]) => void, error?: (e: unknown) => void): void };
}

const isDir = (e: FsEntry): e is FsDirectoryEntry => e.isDirectory;
const isFileEntry = (e: FsEntry): e is FsFileEntry => e.isFile;

function readAll(dir: FsDirectoryEntry): Promise<FsEntry[]> {
  const reader = dir.createReader();
  const all: FsEntry[] = [];
  return new Promise((resolve, reject) => {
    const next = () =>
      reader.readEntries((batch) => {
        if (!batch.length) resolve(all);
        else {
          all.push(...batch);
          next();
        }
      }, reject);
    next();
  });
}

const entryFile = (e: FsFileEntry) => new Promise<File>((resolve, reject) => e.file(resolve, reject));

const clean = (p: string) => p.replace(/^\/+/, "").replace(/\\/g, "/");

/** Recursively collect the files under one entry, keeping relative paths. */
export async function walkEntry(entry: FsEntry): Promise<PickedFile[]> {
  if (isFileEntry(entry)) {
    const file = await entryFile(entry);
    return [{ file, relativePath: clean(entry.fullPath || entry.name) }];
  }
  if (isDir(entry)) {
    const children = await readAll(entry);
    const nested = await Promise.all(children.map(walkEntry));
    return nested.flat();
  }
  return [];
}

interface ItemLike {
  kind: string;
  getAsFile(): File | null;
  webkitGetAsEntry?: () => FsEntry | null;
}

/**
 * Files from a drop. Entries must be taken synchronously, before any await,
 * because the DataTransfer is emptied once the event handler yields.
 */
export async function filesFromDataTransfer(dt: { items?: ArrayLike<ItemLike> | null; files?: ArrayLike<File> | null }): Promise<PickedFile[]> {
  const items = dt.items ? Array.from(dt.items).filter((i) => i.kind === "file") : [];
  if (!items.length) {
    return Array.from(dt.files ?? []).map((file) => ({ file, relativePath: relativePathOf(file) }));
  }
  const pending = items.map((item) => {
    const entry = item.webkitGetAsEntry?.() ?? null;
    if (entry) return walkEntry(entry);
    const file = item.getAsFile();
    return Promise.resolve(file ? [{ file, relativePath: relativePathOf(file) }] : []);
  });
  return (await Promise.all(pending)).flat();
}

/** The relative path of a file from <input webkitdirectory> (or just its name). */
export function relativePathOf(file: File): string {
  const rel = (file as File & { webkitRelativePath?: string; path?: string }).webkitRelativePath;
  return clean(rel || file.name);
}

export function filesFromInput(list: FileList | File[] | null): PickedFile[] {
  return Array.from(list ?? []).map((file) => ({ file, relativePath: relativePathOf(file) }));
}
