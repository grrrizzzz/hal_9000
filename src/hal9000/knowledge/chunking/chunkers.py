"""The four chunkers (plan §6.1) and the profile-driven entry point.

All sizes are model tokens (``tiktoken``, the embedding model's encoding),
measured on a chunk's displayed text: the contextual header never counts
against *N*. Every emitted chunk is verified with the real tokenizer.

* ``fixed``: windows of *N* tokens that snap **back** to the last sentence end
  in the window's final 10% (then a line end, then a word end), so no chunk
  exceeds *N*. The next window starts *O* tokens before the previous end.
* ``recursive``: split on headings, then paragraphs, sentences, lines and
  words (then tokens) until every piece fits; adjacent small pieces are merged
  up to *N*.
* ``structure``: one chunk per logical unit (a heading section, a numbered
  finding with its text and recommendation), top-down: a unit that fits is
  kept whole; one that doesn't yields its own text (split recursively) and its
  sub-units. Tiny units are merged into a neighbour.
* ``parent_child``: parents are structure units packed up to ``parent_size``;
  children of *N* tokens are cut recursively inside each parent.

Overlap *O* in the non-``fixed`` chunkers carries whole trailing sentences
(lines, for text without sentence punctuation) of the previous chunk, within
the same unit or parent, up to *O* tokens. Pieces are split to *N − O* so the
carried sentences fit; a unit that fits in *N* is still kept whole.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import get_args

from hal9000.knowledge.chunking.header import build_header
from hal9000.knowledge.chunking.inputs import ChunkDraft, ChunkInput, Role
from hal9000.knowledge.chunking.text import TextIndex, text_index
from hal9000.knowledge.chunking.units import Node, build_tree, detect_units
from hal9000.knowledge.contracts import ChunkerName, IndexProfile
from hal9000.knowledge.obs.timing import Stage, stage
from hal9000.knowledge.tokens import EMBEDDING_ENCODING, encoding_for_model

__all__ = ["CHUNKERS", "MIN_SIZE", "chunk", "chunk_document"]

CHUNKERS: tuple[str, ...] = get_args(ChunkerName)
#: The smallest *N*: one character is at most 4 tokens, so a window always advances.
MIN_SIZE = 8

Span = tuple[int, int]
Item = tuple[int, int, int]  # a span and the unit it came from
_HEADING, _PARAGRAPH, _SENTENCE, _LINE, _WORD = range(5)


def _room(size: int, overlap: int) -> int:
    """The size to split to so that a carried-over overlap still fits in ``size``."""
    return max(size - overlap, MIN_SIZE // 2)


def chunk_document(inp: ChunkInput, profile: IndexProfile) -> list[ChunkDraft]:
    """Chunk one document as ``profile`` says, timed as the ``chunk`` stage (rule K7)."""
    with stage(Stage.CHUNK):
        return chunk(
            inp,
            profile.chunker,
            size=profile.size,
            overlap=profile.overlap,
            parent_size=profile.parent_size,
            header=profile.header,
            encoding_name=encoding_for_model(profile.embed_model),
        )


def chunk(
    inp: ChunkInput,
    chunker: ChunkerName,
    *,
    size: int,
    overlap: int = 0,
    parent_size: int | None = None,
    header: bool = True,
    encoding_name: str = EMBEDDING_ENCODING,
) -> list[ChunkDraft]:
    """Chunk ``inp`` with one strategy. Output order is document order (a parent precedes its
    children); ``ChunkDraft.key`` is the position in the returned list."""
    if chunker not in CHUNKERS:
        raise ValueError(f"unknown chunker {chunker!r}; expected one of {', '.join(CHUNKERS)}")
    if size < MIN_SIZE:
        raise ValueError(f"size must be at least {MIN_SIZE} tokens, got {size}")
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must be in [0, size), got {overlap} for size {size}")
    if chunker == "parent_child":
        if parent_size is None or parent_size < size:
            raise ValueError("parent_child needs a parent_size of at least the child size")
    elif parent_size is not None:
        raise ValueError("parent_size applies only to the parent_child chunker")
    return _Chunker(inp, encoding_name, header).run(chunker, size, overlap, parent_size or 0)


class _Chunker:
    def __init__(self, inp: ChunkInput, encoding_name: str, header: bool) -> None:
        self.inp = inp
        self.ix: TextIndex = text_index(inp.text, encoding_name)
        units = detect_units(inp.text) if inp.units is None else inp.units
        self.tree = build_tree(units, len(inp.text))
        self.header = header
        self._cuts: list[list[int]] = [
            sorted({n.start for n in self.tree.walk()}),
            self.ix.para_starts,
            self.ix.sentence_starts,
            self.ix.line_starts,
            self.ix.word_starts,
        ]

    def run(self, chunker: str, size: int, overlap: int, parent_size: int) -> list[ChunkDraft]:
        whole = (0, self.ix.length)
        if chunker == "fixed":
            return self._drafts(self._fixed(*whole, size, overlap))
        if chunker == "recursive":
            pieces = self._split(*whole, _HEADING, _room(size, overlap))
            spans = self._with_overlap(pieces, size, overlap)
            return self._drafts(self._merge_tiny(spans, size))
        if chunker == "structure":
            items = self._units(self.tree, size, overlap, group=0)
            return self._drafts(self._merge_tiny([(a, b) for a, b, _ in items], size))
        return self._parent_child(size, overlap, parent_size)

    # ------------------------------------------------------------ helpers
    def _fits(self, a: int, b: int, limit: int) -> bool:
        a, b = self.ix.trim(a, b)
        if b <= a:
            return True
        if self.ix.approx(a, b) > 2 * limit + 64:  # far too big: skip the tokenizer
            return False
        return self.ix.exact(a, b) <= limit

    def _shrink(self, s: int, e: int, limit: int, snap: Callable[[int, int], int]) -> int:
        """Move ``e`` back until ``text[s:e]`` is at most ``limit`` tokens; returns the new end."""
        ix = self.ix
        while True:
            s2, e2 = ix.trim(s, e)
            if e2 <= s2:
                return e
            count = ix.exact(s2, e2)
            if count <= limit or e2 == s2 + 1:  # one code point is at most 4 tokens
                return e2
            t = ix.tokens_before(e2) - (count - limit)
            new_end = snap(s2, ix.token_start(max(t, 0)))
            e = min(max(new_end, s2 + 1), e2 - 1)

    # ---------------------------------------------------------------- fixed
    def _fixed(self, a: int, b: int, limit: int, overlap: int) -> list[Span]:
        ix = self.ix
        tail = max(1, limit // 10)
        spans: list[Span] = []
        s = ix.skip_space(a, b)
        while s < b:
            ts = ix.token_at(s)
            te = ts + limit
            lo = max(s + 1, ix.token_start(max(ts, min(te, ix.n_tokens) - tail)))  # final 10%
            window_end = ix.token_start(te)

            def snap(s2: int, end: int, lo: int = lo) -> int:
                """The last sentence end in ``[lo, end]``, else line end, else word end."""
                for points in (ix.sentence_ends, ix.line_ends, ix.word_ends):
                    found = ix.last_in(points, max(lo, s2 + 1), end)
                    if found is not None:
                        return found
                return end

            e = b if window_end >= b else snap(s, window_end)
            e = self._shrink(s, e, limit, snap)
            spans.append(ix.trim(s, e))
            end = spans[-1][1]
            rest = ix.skip_space(end, b)
            if rest >= b:
                break
            if overlap == 0:
                s = rest
                continue
            back = ix.token_start(max(ix.tokens_before(end) - overlap, ts + 1))
            s = self._word_start(back, s, end, rest)
        return spans

    def _word_start(self, pos: int, prev: int, end: int, rest: int) -> int:
        """A chunk start near ``pos``: back to its word's start, or on to the next word."""
        ix, text = self.ix, self.ix.text
        if pos >= end:
            return rest
        if text[pos].isspace():
            pos = ix.skip_space(pos, end)
        elif pos > 0 and not text[pos - 1].isspace():
            start = ix.last_in(ix.word_starts, prev + 1, pos)
            if start is not None:
                pos = start
        if pos <= prev:
            later = ix.inside(ix.word_starts, prev, end)
            pos = later[0] if later else rest
        return pos

    # ------------------------------------------------------------ recursive
    def _split(self, a: int, b: int, level: int, limit: int) -> list[Span]:
        """Spans covering ``[a, b)``, each at most ``limit`` tokens, split at ``level`` or finer."""
        a, b = self.ix.trim(a, b)
        if b <= a:
            return []
        if self._fits(a, b, limit):
            return [(a, b)]
        for lvl in range(level, len(self._cuts)):
            cuts = self.ix.inside(self._cuts[lvl], a, b)
            if cuts:
                bounds = [a, *cuts, b]
                return self._pack(list(zip(bounds, bounds[1:])), lvl, limit)
        return self._hard(a, b, limit)

    def _pack(self, pieces: list[Span], level: int, limit: int) -> list[Span]:
        """Greedily merge adjacent pieces up to ``limit``; split oversized pieces finer."""
        ix = self.ix
        out: list[Span] = []
        buf: list[Span] = []
        for piece in pieces:
            pa, pb = ix.trim(*piece)
            if pb <= pa:
                continue
            if ix.approx(pa, pb) > limit:
                out += self._emit(buf, level, limit)
                buf = []
                out += self._split(pa, pb, level + 1, limit)
            elif buf and ix.approx(buf[0][0], pb) <= limit:
                buf.append((pa, pb))
            else:
                out += self._emit(buf, level, limit)
                buf = [(pa, pb)]
        return out + self._emit(buf, level, limit)

    def _emit(self, buf: list[Span], level: int, limit: int) -> list[Span]:
        """Emit buffered pieces as chunks, checking each with the real tokenizer."""
        out: list[Span] = []
        while buf:
            k = len(buf)
            while k and not self._fits(buf[0][0], buf[k - 1][1], limit):
                k -= 1
            if k == 0:
                out += self._split(*buf[0], level + 1, limit)
                k = 1
            else:
                out.append(self.ix.trim(buf[0][0], buf[k - 1][1]))
            buf = buf[k:]
        return out

    def _hard(self, a: int, b: int, limit: int) -> list[Span]:
        """Token windows for text with no usable split point (one long 'word')."""
        ix = self.ix
        spans: list[Span] = []
        s = ix.skip_space(a, b)
        while s < b:
            e = min(b, ix.token_start(ix.token_at(s) + limit))
            e = self._shrink(s, max(e, s + 1), limit, lambda _s, end: end)
            spans.append(ix.trim(s, e))
            s = ix.skip_space(spans[-1][1], b)
        return spans

    def _with_overlap(self, spans: list[Span], limit: int, overlap: int) -> list[Span]:
        """Start each chunk up to ``overlap`` tokens early, at a sentence (else line) start."""
        if overlap == 0 or len(spans) < 2:
            return spans
        ix = self.ix
        out = [spans[0]]
        for cs, ce in spans[1:]:
            ps, pe = out[-1]
            start = cs
            for points in (ix.sentence_starts, ix.line_starts):
                found = self._overlap_start(points, ps, pe, ce, limit, overlap)
                if found is not None:
                    start = found
                    break
            out.append((start, ce))
        return out

    def _overlap_start(
        self, points: list[int], ps: int, pe: int, ce: int, limit: int, overlap: int
    ) -> int | None:
        """The earliest point inside the previous chunk ``[ps, pe)`` whose tail fits in
        ``overlap`` tokens while the extended chunk ``[point, ce)`` fits in ``limit``."""
        ix = self.ix
        cands = ix.inside(points, ps, pe)
        lo, hi = 0, len(cands)
        while lo < hi:  # the tail's token count falls as the candidate moves right
            mid = (lo + hi) // 2
            if ix.approx(cands[mid], pe) <= overlap:
                hi = mid
            else:
                lo = mid + 1
        for q in cands[lo : lo + 4]:
            if ix.exact(q, pe) <= overlap and ix.exact(q, ce) <= limit:
                return q
        return None

    # ------------------------------------------------------------ structure
    def _units(self, node: Node, limit: int, overlap: int, group: int) -> list[Item]:
        """Top-down: a unit that fits is one chunk; otherwise its own text and its sub-units."""
        a, b = self.ix.trim(node.start, node.end)
        if b <= a:
            return []
        if self._fits(a, b, limit):
            return [(a, b, group)]
        items: list[Item] = []
        me = id(node)

        def gap(start: int, end: int) -> None:
            spans = self._split(start, end, _PARAGRAPH, _room(limit, overlap))
            items.extend((s, e, me) for s, e in self._with_overlap(spans, limit, overlap))

        pos = node.start
        for child in node.children:
            if pos < child.start:
                gap(pos, child.start)
            items += self._units(child, limit, overlap, me)
            pos = child.end
        if pos < node.end:
            gap(pos, node.end)
        return items

    def _merge_tiny(self, spans: list[Span], limit: int) -> list[Span]:
        """Merge chunks under ``min(32, N/4)`` tokens into the next chunk, else the previous."""
        tiny = min(32, limit // 4)
        out: list[Span] = []
        carry: Span | None = None

        def absorb(item: Span) -> None:
            if out and self._fits(out[-1][0], item[1], limit):
                out[-1] = (out[-1][0], item[1])
            else:
                out.append(item)

        for span in spans:
            if carry is not None:
                if self._fits(carry[0], span[1], limit):
                    span = (carry[0], span[1])
                else:
                    absorb(carry)
                carry = None
            if self.ix.exact(*span) < tiny:
                carry = span
            else:
                out.append(span)
        if carry is not None:
            absorb(carry)
        return out

    # --------------------------------------------------------- parent_child
    def _parent_child(self, size: int, overlap: int, parent_size: int) -> list[ChunkDraft]:
        items = self._units(self.tree, parent_size, 0, group=0)
        runs: list[list[Item]] = []
        for item in items:  # sibling units, packed by the cheap count; checked below
            run = runs[-1] if runs else None
            if run and run[0][2] == item[2] and self.ix.approx(run[0][0], item[1]) <= parent_size:
                run.append(item)
            else:
                runs.append([item])
        packed: list[Span] = []
        for run in runs:
            packed += self._emit([(a, b) for a, b, _ in run], _WORD, parent_size)
        parents = self._merge_tiny(packed, parent_size)
        drafts: list[ChunkDraft] = []
        for pa, pb in parents:
            parent_key = len(drafts)
            drafts.append(self._draft(parent_key, pa, pb, "parent", None))
            pieces = self._split(pa, pb, _HEADING, _room(size, overlap))
            children = self._with_overlap(pieces, size, overlap)
            for ca, cb in children:
                drafts.append(self._draft(len(drafts), ca, cb, "child", parent_key))
        return drafts

    # --------------------------------------------------------------- output
    def _drafts(self, spans: list[Span]) -> list[ChunkDraft]:
        return [self._draft(i, a, b, "chunk", None) for i, (a, b) in enumerate(spans)]

    def _draft(self, key: int, a: int, b: int, role: Role, parent_key: int | None) -> ChunkDraft:
        inp = self.inp
        path = self.tree.path_at(a)
        header = (
            build_header(inp.title, inp.doc_type, inp.vendor, inp.doc_date, path)
            if self.header
            else ""
        )
        return ChunkDraft(
            key=key,
            document_id=inp.document_id,
            char_start=a,
            char_end=b,
            text=inp.text[a:b],
            header=header,
            tokens=self.ix.exact(a, b),
            page_start=inp.page_of(a),
            page_end=inp.page_of(b - 1),
            section_path=path,
            role=role,
            parent_key=parent_key,
        )
