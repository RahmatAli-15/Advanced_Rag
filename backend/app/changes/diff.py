"""Paragraph-level diff between two versions of a regulation.

Compliance teams ask "what changed?", not "which bytes changed?". Both versions are
parsed and chunked with the same clause-aware chunker used for search, then compared
paragraph by paragraph (keyed by paragraph reference, e.g. "42(1)"). Whitespace-only
differences are ignored.
"""

import re
from collections.abc import Callable, Sequence
from typing import Any

from app.chunking.chunker import ChunkDraft, chunk_blocks
from app.ingestion.encoding import decode_html
from app.ingestion.sources import rbi

MAX_CHANGED = 40  # keep alerts readable; counts still cover everything
EXCERPT = 600

Parser = Callable[[str], rbi.DetailPage]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def paragraphs(drafts: Sequence[ChunkDraft]) -> dict[str, str]:
    """Paragraph ref -> its full text (a clause split across chunks is joined back).
    Unnumbered text is keyed by its heading."""
    out: dict[str, str] = {}
    for d in drafts:
        key = d.para_refs[0] if d.para_refs else "§ " + (" > ".join(d.heading_path) or "preamble")
        text = re.sub(r"\s+", " ", d.text).strip()
        out[key] = f"{out[key]} {text}" if key in out else text
    return out


def diff_paragraphs(old: dict[str, str], new: dict[str, str]) -> dict[str, Any]:
    added = [k for k in new if k not in old]
    removed = [k for k in old if k not in new]
    changed = [k for k in new if k in old and _norm(old[k]) != _norm(new[k])]
    return {
        "counts": {"added": len(added), "removed": len(removed), "changed": len(changed)},
        "added": [{"para": k, "after": new[k][:EXCERPT]} for k in added[:MAX_CHANGED]],
        "removed": [{"para": k, "before": old[k][:EXCERPT]} for k in removed[:MAX_CHANGED]],
        "changed": [
            {"para": k, "before": old[k][:EXCERPT], "after": new[k][:EXCERPT]}
            for k in changed[:MAX_CHANGED]
        ],
    }


def diff_raw(
    old_raw: bytes | None, new_raw: bytes, parse: Parser = rbi.parse_detail
) -> dict[str, Any]:
    """Diff two stored pages. With no old text (a new document, or only the version
    date of the previous text is known), everything is reported as added."""
    new = paragraphs(chunk_blocks(parse(decode_html(new_raw)).blocks))
    old = paragraphs(chunk_blocks(parse(decode_html(old_raw)).blocks)) if old_raw else {}
    return diff_paragraphs(old, new)
