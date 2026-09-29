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

/** Shortest block (normalised characters) highlighted on its own. */
export const MIN_BLOCK = 24;

/**
 * Locate `needle` in `haystack` (both already normalised). A chunk isn't
 * always one contiguous run on the page (a running header or footer that
 * cleaning removed can sit in the middle, or columns interleave), so every
 * aligned block of at least MIN_BLOCK characters is returned, in page order.
 */
export function locateAll(haystack: string, needle: string, minScore = 0.35, minBlock = MIN_BLOCK): Match[] {
  if (!needle || !haystack) return [];
  const exact = haystack.indexOf(needle);
  if (exact >= 0) return [{ start: exact, end: exact + needle.length, score: 1 }];

  const n = needle.length < 24 ? 4 : 8;
  if (needle.length < n) return [];
  const index = new Map<string, number[]>();
  for (let i = 0; i + n <= haystack.length; i++) {
    const g = haystack.slice(i, i + n);
    const list = index.get(g);
    if (list) list.push(i);
    else index.set(g, [i]);
  }
  // Each shared n-gram votes for an alignment (needle offset → page offset), bucketed to absorb small shifts.
  const bucket = 6;
  const byBucket = new Map<number, { h: number; k: number }[]>();
  let grams = 0;
  for (let k = 0; k + n <= needle.length; k++) {
    grams++;
    const positions = index.get(needle.slice(k, k + n));
    if (!positions || positions.length > 8) continue;
    for (const h of positions) {
      const b = Math.round((h - k) / bucket);
      const list = byBucket.get(b) ?? [];
      list.push({ h, k });
      byBucket.set(b, list);
    }
  }
  if (!byBucket.size) return [];

  // Adjacent alignment buckets form one block candidate.
  const keys = [...byBucket.keys()].sort((a, b) => a - b);
  const blocks: { start: number; end: number; votes: number }[] = [];
  let group: { h: number; k: number }[] = [];
  const flush = () => {
    if (!group.length) return;
    // Within one alignment, split where the page positions jump (the same shift can recur far away).
    group.sort((a, b) => a.h - b.h);
    let cur = { start: group[0].h, end: group[0].h + n, votes: 0 };
    for (const { h } of group) {
      if (h > cur.end + n) {
        blocks.push(cur);
        cur = { start: h, end: h + n, votes: 0 };
      }
      cur.end = Math.max(cur.end, h + n);
      cur.votes++;
    }
    blocks.push(cur);
    group = [];
  };
  keys.forEach((b, i) => {
    if (i && b - keys[i - 1] > 1) flush();
    group.push(...byBucket.get(b)!);
  });
  flush();

  // Keep dense, long-enough blocks: a real aligned run of L characters yields about L - n + 1 votes.
  const good = blocks.filter((bl) => bl.end - bl.start >= Math.min(minBlock, needle.length) && bl.votes >= 0.5 * (bl.end - bl.start - n + 1));
  const covered = good.reduce((sum, bl) => sum + bl.votes, 0);
  if (!good.length || covered / grams < minScore) return [];
  // Merge overlaps, in page order.
  good.sort((a, b) => a.start - b.start);
  const out: Match[] = [];
  for (const bl of good) {
    const last = out[out.length - 1];
    if (last && bl.start <= last.end) last.end = Math.max(last.end, bl.end);
    else out.push({ start: bl.start, end: bl.end, score: covered / grams });
  }
  return out;
}

/** The largest aligned block (see locateAll). */
export function locate(haystack: string, needle: string, minScore = 0.35): Match | null {
  const all = locateAll(haystack, needle, minScore);
  return all.length ? all.reduce((a, b) => (b.end - b.start > a.end - a.start ? b : a)) : null;
}

/** Per text item, the [from, to) character ranges to highlight. */
export type ItemRanges = Map<number, [number, number][]>;

export function highlightRanges(items: string[], needles: string[]): ItemRanges {
  const hay = normalizeItems(items);
  const ranges: ItemRanges = new Map();
  for (const raw of needles) {
    for (const m of locateAll(hay.text, normalize(raw))) {
      for (let p = m.start; p < m.end; p++) {
        const [item, ch] = hay.map[p];
        const list = ranges.get(item) ?? [];
        const last = list[list.length - 1];
        // Extend across non-alphanumerics (spaces, punctuation) inside the match.
        if (last && ch >= last[0] && ch <= last[1] + 3) last[1] = Math.max(last[1], ch + 1);
        else list.push([ch, ch + 1]);
        ranges.set(item, list);
      }
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
    for (const m of locateAll(hay, normalize(raw))) out.push([map[m.start][1], map[m.end - 1][1] + 1]);
  }
  return out.sort((a, b) => a[0] - b[0]);
}
