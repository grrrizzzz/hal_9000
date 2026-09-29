/** Split answer text into plain runs and `[S#]` citation markers. */
export type Segment = { kind: "text"; text: string } | { kind: "cite"; marker: string };

const MARKER = /\[(S[1-9]\d*)(?:\s*[,;]\s*(S[1-9]\d*))*\]/g;

export function splitCitations(text: string): Segment[] {
  const out: Segment[] = [];
  let last = 0;
  for (const m of text.matchAll(MARKER)) {
    const idx = m.index ?? 0;
    if (idx > last) out.push({ kind: "text", text: text.slice(last, idx) });
    // "[S1, S3]" becomes two chips.
    for (const marker of m[0].slice(1, -1).split(/\s*[,;]\s*/)) out.push({ kind: "cite", marker });
    last = idx + m[0].length;
  }
  if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
  return out;
}
