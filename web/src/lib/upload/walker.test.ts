import { describe, expect, it } from "vitest";
import { filesFromDataTransfer, walkEntry, type FsDirectoryEntry, type FsEntry, type FsFileEntry } from "./walker";

// Fakes for the (non-standard, jsdom-less) FileSystem entry API.
function fileEntry(fullPath: string, body = "x"): FsFileEntry {
  const name = fullPath.split("/").pop()!;
  return {
    isFile: true,
    isDirectory: false,
    name,
    fullPath,
    file: (ok) => setTimeout(() => ok(new File([body], name)), 0),
  };
}

/** A directory whose reader hands out entries in batches, like Chrome's (≤100 per call). */
function dirEntry(fullPath: string, children: FsEntry[], batch = 2): FsDirectoryEntry & { reads: number } {
  const dir = {
    isFile: false,
    isDirectory: true,
    name: fullPath.split("/").pop()!,
    fullPath,
    reads: 0,
    createReader() {
      let i = 0;
      return {
        readEntries(ok: (e: FsEntry[]) => void) {
          dir.reads++;
          const out = children.slice(i, i + batch);
          i += batch;
          setTimeout(() => ok(out), 0);
        },
      };
    },
  };
  return dir;
}

describe("walkEntry", () => {
  it("walks nested folders and keeps relative paths", async () => {
    const furnace = dirEntry("/House/Appliances/Furnace", [
      fileEntry("/House/Appliances/Furnace/manual.pdf"),
      fileEntry("/House/Appliances/Furnace/warranty.docx"),
      fileEntry("/House/Appliances/Furnace/photo.jpg"),
    ]);
    const appliances = dirEntry("/House/Appliances", [furnace, fileEntry("/House/Appliances/list.md")]);
    const root = dirEntry("/House", [fileEntry("/House/inspection.pdf"), appliances, dirEntry("/House/Empty", [])]);

    const files = await walkEntry(root);
    const paths = files.map((f) => f.relativePath).sort();

    expect(paths).toEqual([
      "House/Appliances/Furnace/manual.pdf",
      "House/Appliances/Furnace/photo.jpg",
      "House/Appliances/Furnace/warranty.docx",
      "House/Appliances/list.md",
      "House/inspection.pdf",
    ]);
    expect(files.every((f) => f.file instanceof File)).toBe(true);
    expect(files.find((f) => f.relativePath.endsWith("manual.pdf"))!.file.name).toBe("manual.pdf");
  });

  it("keeps calling readEntries until a directory returns an empty batch", async () => {
    const many = Array.from({ length: 7 }, (_, i) => fileEntry(`/Big/f${i}.txt`));
    const big = dirEntry("/Big", many, 3);
    const files = await walkEntry(big);
    expect(files).toHaveLength(7);
    expect(big.reads).toBe(4); // 3 + 3 + 1, then the empty batch
  });
});

describe("filesFromDataTransfer", () => {
  it("mixes dropped folders and loose files", async () => {
    const folder = dirEntry("/Receipts", [fileEntry("/Receipts/2023/roof.pdf"), fileEntry("/Receipts/a.txt")]);
    const loose = new File(["hi"], "notes.md");
    const items = [
      { kind: "file", getAsFile: () => null, webkitGetAsEntry: () => folder },
      // Some browsers (and synthetic drops) have no entry: fall back to the File.
      { kind: "file", getAsFile: () => loose, webkitGetAsEntry: () => null },
      { kind: "string", getAsFile: () => null },
    ];
    const files = await filesFromDataTransfer({ items });
    expect(files.map((f) => f.relativePath).sort()).toEqual(["Receipts/2023/roof.pdf", "Receipts/a.txt", "notes.md"]);
  });

  it("uses dataTransfer.files when there are no items", async () => {
    const f = new File(["x"], "a.pdf");
    const files = await filesFromDataTransfer({ files: [f] });
    expect(files).toEqual([{ file: f, relativePath: "a.pdf" }]);
  });
});
