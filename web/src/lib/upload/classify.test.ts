import { describe, expect, it } from "vitest";
import { summarize } from "./classify";

const pick = (relativePath: string, size = 10) => ({ relativePath, file: new File(["x".repeat(size)], relativePath.split("/").pop()!) });

describe("summarize", () => {
  it("counts what will be added and groups skipped files by extension", () => {
    const s = summarize([
      pick("House/report.pdf"),
      pick("House/manual.DOCX"),
      pick("House/notes.txt"),
      pick("House/readme.md"),
      pick("House/page.html"),
      ...Array.from({ length: 5 }, (_, i) => pick(`House/img/p${i}.jpg`)),
      pick("House/img/a.heic"),
      pick("House/img/b.heic"),
      pick("House/.DS_Store"),
      pick("House/data.xlsx"),
    ]);
    expect(s.added.map((a) => a.relativePath)).toHaveLength(5);
    expect(s.text).toBe("14 files found: 5 will be added, 9 skipped (.jpg ×5, .heic ×2, .xlsx ×1, hidden ×1)");
    expect(s.hasImages).toBe(true);
  });

  it("skips empty files and says so in the singular", () => {
    const s = summarize([pick("empty.pdf", 0)]);
    expect(s.text).toBe("1 file found: 0 will be added, 1 skipped (.pdf ×1)");
    expect(s.skipped[0].skip).toBe("empty");
  });

  it("counts a file picked twice once", () => {
    const s = summarize([pick("a.pdf"), pick("a.pdf")]);
    expect(s.total).toBe(1);
  });
});
