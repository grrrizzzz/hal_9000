/**
 * Placing a highlight on a pdf.js text layer.
 *
 * Chunk and evidence text come from *cleaned* page text (headers removed,
 * hyphenation repaired, letter-spaced headings rejoined), so their offsets
 * don't line up with pdf.js's raw text items (plan §10). We normalise both
 * sides to lowercase alphanumerics, map every normalised character back to
 * (item, offset), and locate the needle: an exact match when there is one,
 * otherwise by voting on the alignment of shared n-grams, which tolerates
 * small differences (a dropped header line, a rejoined hyphen).
 */

export interface NormText {
  text: string;
  /** For each character of `text`, its [itemIndex, charIndex] in the source. */
  map: [number, number][];
}

const WORDISH = /[\p{L}\p{N}]/u;

export function normalizeItems(items: string[]): NormText {
  let text = "";
  const map: [number, number][] = [];
  items.forEach((s, i) => {
    let j = 0;
    for (const ch of s) {
      if (WORDISH.test(ch)) {
        for (const lower of ch.toLowerCase()) {
          text += lower;
          map.push([i, j]);
        }
      }
      j += ch.length;
    }
  });
  return { text, map };
}

export function normalize(s: string): string {
  return normalizeItems([s]).text;
}

export interface Match {
  start: number; // inclusive, in normalised haystack
  end: number; // exclusive
  score: number; // share of needle n-grams that agreed (1 for exact)
}

/** Locate `needle` in `haystack` (both already normalised). */
export function locate(haystack: string, needle: string, minScore = 0.35): Match | null {
  if (!needle || !haystack) return null;
  const exact = haystack.indexOf(needle);
  if (exact >= 0) return { start: exact, end: exact + needle.length, score: 1 };

  const n = needle.length < 24 ? 4 : 8;
  if (needle.length < n) return null;
  const index = new Map<string, number[]>();
  for (let i = 0; i + n <= haystack.length; i++) {
    const g = haystack.slice(i, i + n);
    const list = index.get(g);
    if (list) list.push(i);
    else index.set(g, [i]);
  }
  // Vote on the needle's start position in the haystack (bucketed to absorb small shifts).
  const bucket = 6;
  const votes = new Map<number, number>();
  const hits: { h: number; k: number }[] = [];
  let grams = 0;
  for (let k = 0; k + n <= needle.length; k += 1) {
    grams++;
    const positions = index.get(needle.slice(k, k + n));
    if (!positions || positions.length > 8) continue;
    for (const h of positions) {
      const b = Math.round((h - k) / bucket);
      votes.set(b, (votes.get(b) ?? 0) + 1);
      hits.push({ h, k });
    }
  }
  if (!votes.size) return null;
  let best = 0;
  let bestVotes = 0;
  for (const [b, v] of votes) {
    const neighbourhood = v + (votes.get(b - 1) ?? 0) + (votes.get(b + 1) ?? 0);
    if (neighbourhood > bestVotes) {
      bestVotes = neighbourhood;
      best = b;
    }
  }
  const score = bestVotes / grams;
  if (score < minScore) return null;
  let start = Infinity;
  let end = -Infinity;
  for (const { h, k } of hits) {
    if (Math.abs(Math.round((h - k) / bucket) - best) <= 1) {
      start = Math.min(start, h);
      end = Math.max(end, h + n);
    }
  }
  return { start, end, score };
}

/** Per text item, the [from, to) character ranges to highlight. */
export type ItemRanges = Map<number, [number, number][]>;

export function highlightRanges(items: string[], needles: string[]): ItemRanges {
  const hay = normalizeItems(items);
  const ranges: ItemRanges = new Map();
  for (const raw of needles) {
    const needle = normalize(raw);
    const m = locate(hay.text, needle);
    if (!m) continue;
    for (let p = m.start; p < m.end; p++) {
      const [item, ch] = hay.map[p];
      const list = ranges.get(item) ?? [];
      const last = list[list.length - 1];
      // Extend across non-alphanumerics (spaces, punctuation) inside the match.
      if (last && ch <= last[1] + 3) last[1] = ch + 1;
      else list.push([ch, ch + 1]);
      ranges.set(item, list);
    }
  }
  return ranges;
}

const ESC: Record<string, string> = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
export const escapeHtml = (s: string) => s.replace(/[&<>"']/g, (c) => ESC[c]);

/** Render one text item as HTML with <mark>s (pdf.js sets it as innerHTML, so everything is escaped). */
export function renderItem(str: string, ranges: [number, number][] | undefined): string {
  if (!ranges?.length) return escapeHtml(str);
  let out = "";
  let pos = 0;
  for (const [a, b] of ranges) {
    out += escapeHtml(str.slice(pos, a));
    out += `<mark data-hl="1">${escapeHtml(str.slice(a, b))}</mark>`;
    pos = b;
  }
  return out + escapeHtml(str.slice(pos));
}

/** Character ranges in plain (cleaned) text for the text viewer. */
export function textRanges(text: string, needles: string[]): [number, number][] {
  const { text: hay, map } = normalizeItems([text]);
  const out: [number, number][] = [];
  for (const raw of needles) {
    const m = locate(hay, normalize(raw));
    if (!m) continue;
    out.push([map[m.start][1], map[m.end - 1][1] + 1]);
  }
  return out.sort((a, b) => a[0] - b[0]);
}
