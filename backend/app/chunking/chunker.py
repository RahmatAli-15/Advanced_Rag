"""Clause-aware chunking for regulatory documents.

Why not fixed-size chunks? A regulation's meaning lives in its structure. A sub-clause
like "(b) within 30 days" is useless without "5. A lender shall refund the fee ...".

Algorithm
1. Walk blocks, tracking the heading path (Chapter > Section) and clause numbering.
2. Every numbered paragraph starts an *item*; Explanation / Provided / Note and other
   unnumbered text continue the current item.
3. Items under the same top-level clause (e.g. 5, 5(1), 5(1)(i)) form a *clause*.
4. A clause that fits the token budget is one chunk. A larger clause is packed into
   several chunks at item boundaries, and each continuation chunk starts with the
   clause's lead-in so it still makes sense alone. A single oversized item is split
   by sentences with a one-sentence overlap.
5. Consecutive tiny clauses in the same section are merged so chunks carry context.
6. Tables become their own chunks, split by rows with the header row repeated.

Every chunk records its exact paragraph references and the amendments that touched
it, so answers can cite "para 5(2)(iv), as substituted w.e.f. 1 July 2026".
"""

import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

from app.chunking.numbering import Marker, NumberingTracker, detect_marker, render_ref
from app.ingestion.document import Block, Footnote

CHUNKER_VERSION = "clause-v1"

TokenCounter = Callable[[str], int]
ChunkKind = Literal["text", "table"]


def approx_tokens(text: str) -> int:
    """~4 characters per token for English legal text. Swapped for the embedding
    model's real tokenizer once one is chosen."""
    return math.ceil(len(text) / 4)


@dataclass(frozen=True)
class ChunkConfig:
    max_tokens: int = 400  # leaves room for the context header in a 512-token model
    min_tokens: int = 60  # clauses smaller than this get merged with neighbours
    lead_in_tokens: int = 60  # how much of the clause intro to repeat in continuations


@dataclass(frozen=True)
class ChunkDraft:
    ordinal: int
    kind: ChunkKind
    heading_path: tuple[str, ...]
    para_refs: tuple[str, ...]
    lead_in: str | None  # clause intro repeated for context; not part of `text`
    text: str
    footnote_ids: tuple[str, ...]
    token_count: int

    def context_header(self, doc_title: str) -> str:
        """What the embedding model and keyword index see in front of the text."""
        parts = [doc_title, *self.heading_path]
        if self.para_refs:
            parts.append("Para " + ", ".join(self.para_refs[:6]))
        header = " > ".join(parts)
        return f"{header}\n{self.lead_in} …" if self.lead_in else header


@dataclass
class _Item:
    ref: str
    texts: list[str]
    footnote_ids: list[str]
    top: str  # top-level clause key, "" for unnumbered preamble text
    ancestors: tuple[str, ...] = ()  # refs of enclosing items: 5(1)(xiv)(d) -> 5, 5(1), 5(1)(xiv)

    @property
    def text(self) -> str:
        return "\n".join(self.texts)


@dataclass
class _Section:
    """A run of content under one heading path: clauses (lists of items) and tables."""

    heading_path: tuple[str, ...]
    parts: list[list[_Item] | Block] = field(default_factory=list)


_TOP_HEADING = re.compile(r"^(Chapter|Part|Annex|Annexure|Schedule|Appendix)\b", re.I)


def _sections(blocks: Iterable[Block]) -> list[_Section]:
    sections: list[_Section] = []
    level0: str | None = None
    level1: str | None = None
    tracker = NumberingTracker()
    current: _Section | None = None

    def section() -> _Section:
        nonlocal current
        path = tuple(h for h in (level0, level1) if h)
        if current is None or current.heading_path != path:
            current = _Section(path)
            sections.append(current)
        return current

    for block in blocks:
        if block.kind == "heading":
            if _TOP_HEADING.match(block.text):
                level0, level1 = block.text, None
            else:
                level1 = block.text
            tracker.reset()
            # Headings like "63. CDD Procedure" carry the clause number their
            # sub-items hang off: "(9)" beneath it is really "63(9)".
            heading_marker = detect_marker(block.text)
            if heading_marker is not None and heading_marker.style == "decimal":
                tracker.advance(block.text)
            continue

        sec = section()
        if block.kind == "table":
            sec.parts.append(block)
            continue

        marker = tracker.advance(block.text)
        clause = sec.parts[-1] if sec.parts and isinstance(sec.parts[-1], list) else None
        if marker is None:
            if clause:  # Explanation / Provided / Note / wrapped text
                clause[-1].texts.append(block.text)
                clause[-1].footnote_ids.extend(block.footnote_ids)
            else:
                sec.parts.append([_Item("", [block.text], list(block.footnote_ids), "")])
            continue

        path: tuple[Marker, ...] = tracker.path
        top = path[0].render() if path else ""
        ancestors = tuple(render_ref(path[:i]) for i in range(1, len(path)))
        item = _Item(render_ref(path), [block.text], list(block.footnote_ids), top, ancestors)
        if clause and clause[-1].top == top and top and marker.style != "decimal":
            clause.append(item)
        else:
            sec.parts.append([item])
    return sections


def _split_words(text: str, budget: int, count: TokenCounter) -> list[str]:
    """Last resort for text with no usable sentence breaks (e.g. long lists of Acts)."""
    pieces: list[str] = []
    buf: list[str] = []
    for word in text.split():
        if buf and count(" ".join([*buf, word])) > budget:
            pieces.append(" ".join(buf))
            buf = []
        buf.append(word)
    if buf:
        pieces.append(" ".join(buf))
    return pieces


def _split_sentences(text: str, budget: int, count: TokenCounter) -> list[str]:
    sentences = re.split(r"(?<=[.;:])\s+(?=[A-Z(\[])", text)
    pieces: list[str] = []
    buf: list[str] = []
    for s in sentences:
        if buf and count(" ".join([*buf, s])) > budget:
            pieces.append(" ".join(buf))
            # one-sentence overlap keeps the thread, unless that sentence is itself huge
            buf = [buf[-1]] if count(buf[-1]) < budget // 3 else []
        buf.append(s)
    if buf:
        pieces.append(" ".join(buf))
    return [
        p
        for piece in pieces
        for p in (_split_words(piece, budget, count) if count(piece) > budget else [piece])
    ]


def _truncate(text: str, tokens: int, count: TokenCounter) -> str:
    if count(text) <= tokens:
        return text
    words = text.split()
    while words and count(" ".join(words)) > tokens:
        words = words[: max(1, int(len(words) * 0.8))]
    return " ".join(words)


def _lead_in(chain: list[_Item], budget: int, count: TokenCounter) -> str | None:
    if not chain:
        return None
    share = max(20, budget // len(chain))
    return " … ".join(_truncate(i.texts[0], share, count) for i in chain)


@dataclass
class _Pending:
    refs: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    footnotes: list[str] = field(default_factory=list)
    lead_in: str | None = None


def chunk_blocks(
    blocks: list[Block],
    config: ChunkConfig | None = None,
    count: TokenCounter = approx_tokens,
) -> list[ChunkDraft]:
    cfg = config or ChunkConfig()
    drafts: list[ChunkDraft] = []

    def emit(kind: ChunkKind, path: tuple[str, ...], p: _Pending) -> None:
        text = "\n".join(p.texts).strip()
        if not text:
            return
        drafts.append(
            ChunkDraft(
                ordinal=len(drafts),
                kind=kind,
                heading_path=path,
                para_refs=tuple(dict.fromkeys(r for r in p.refs if r)),
                lead_in=p.lead_in,
                text=text,
                footnote_ids=tuple(dict.fromkeys(p.footnotes)),
                token_count=count(text),
            )
        )

    for sec in _sections(blocks):
        small = _Pending()  # accumulates tiny clauses for merging

        def flush_small(sec: _Section = sec, small: _Pending = small) -> None:
            if small.texts:
                emit("text", sec.heading_path, _Pending(small.refs, small.texts, small.footnotes))
                small.refs, small.texts, small.footnotes = [], [], []

        for part in sec.parts:
            if isinstance(part, Block):  # table
                flush_small()
                for rows in _table_pieces(part.text, cfg.max_tokens, count):
                    emit("table", sec.heading_path, _Pending([], [rows], list(part.footnote_ids)))
                continue

            clause_text = "\n".join(i.text for i in part)
            clause_tokens = count(clause_text)
            if clause_tokens < cfg.min_tokens:
                if small.texts and count("\n".join([*small.texts, clause_text])) > cfg.max_tokens:
                    flush_small()
                small.refs.extend(i.ref for i in part)
                small.texts.append(clause_text)
                small.footnotes.extend(f for i in part for f in i.footnote_ids)
                continue
            flush_small()

            if clause_tokens <= cfg.max_tokens:
                emit(
                    "text",
                    sec.heading_path,
                    _Pending(
                        [i.ref for i in part],
                        [clause_text],
                        [f for i in part for f in i.footnote_ids],
                    ),
                )
                continue

            # Oversized clause: pack items; each continuation chunk starts with the
            # opening lines of the items it hangs off (5 > (1) > (xiv)), so a chunk
            # starting at "(d) ..." still says what (d) is part of.
            by_ref = {i.ref: i for i in part if i.ref}
            cur = _Pending()
            for item in part:
                pieces = (
                    [item.text]
                    if count(item.text) <= cfg.max_tokens
                    else _split_sentences(item.text, cfg.max_tokens, count)
                )
                for n, piece in enumerate(pieces):
                    if cur.texts and count("\n".join([*cur.texts, piece])) > cfg.max_tokens:
                        emit("text", sec.heading_path, cur)
                        chain = [by_ref[r] for r in item.ancestors if r in by_ref]
                        if n > 0:
                            chain.append(item)  # continuing a split item: repeat its start
                        if not chain and part[0] is not item:
                            chain = [part[0]]
                        cur = _Pending(lead_in=_lead_in(chain, cfg.lead_in_tokens, count))
                    cur.refs.append(item.ref)
                    cur.texts.append(piece)
                    cur.footnotes.extend(item.footnote_ids)
            emit("text", sec.heading_path, cur)
        flush_small()
    return drafts


def _table_pieces(table_text: str, budget: int, count: TokenCounter) -> list[str]:
    rows = table_text.split("\n")
    header, body = rows[0], rows[1:]
    if count(table_text) <= budget or not body:
        return [table_text]
    row_budget = max(budget - count(header), budget // 2)
    pieces: list[str] = []
    cur = [header]
    for row in body:
        # A single row can hold a whole page of text; split it, keeping the header.
        for part in [row] if count(row) <= row_budget else _split_sentences(row, row_budget, count):
            if len(cur) > 1 and count("\n".join([*cur, part])) > budget:
                pieces.append("\n".join(cur))
                cur = [header]
            cur.append(part)
    pieces.append("\n".join(cur))
    return pieces


def resolve_amendments(
    draft: ChunkDraft, footnotes: dict[str, Footnote]
) -> list[dict[str, str | None]]:
    """Footnote ids -> JSON-ready amendment records for storage and citation."""
    out: list[dict[str, str | None]] = []
    for fid in draft.footnote_ids:
        fn = footnotes.get(fid)
        if fn is None:
            continue
        out.append(
            {
                "id": fn.id,
                "action": fn.action,
                "effective_from": fn.effective_from.isoformat() if fn.effective_from else None,
                "amending_title": fn.amending_title,
                "amending_url": fn.amending_url,
                "text": fn.text,
            }
        )
    return out
