# ADR 0004: Clause-aware chunking

**Status:** Accepted · **Date:** 2026-11-18

## Context

Fixed-size chunking (every N tokens) cuts regulations mid-clause. A retrieved chunk
"(b) within 30 days of the request" is useless without its parent "5. A lender shall
refund the processing fee …", and the answer cannot cite which paragraph it came from.

Measured on the 45 NBFC Master Directions (Nov 2026): paragraphs are short (median 22
words) but some are long (max 1,111 words); numbering nests up to four levels in a
document-specific order (`5.` › `(1)` › `(i)` › `(a)`, or `(iv)` › `(a)`); 139
footnotes record amendments, each linked from the exact paragraph it changed.

## Decision

1. **Structure first.** Chunk boundaries follow headings and clause numbering, not
   token counts. A numbering tracker learns each document's hierarchy from the order
   styles appear, and resolves `(i)`/`(v)`/`(x)` as letters only when they continue
   an alphabetic run.
2. **One clause per chunk when it fits** (≤ 400 tokens, leaving room for a header in a
   512-token embedding model). Explanation / Provided / Note stay with their item.
3. **Oversized clauses split at item boundaries.** Each continuation chunk carries a
   *lead-in*: the opening lines of every ancestor item (`5. … (1) … (xiv) 'OVD' means`),
   so it stands alone. A single huge item splits by sentences with one sentence of
   overlap, then by words as a last resort.
4. **Tiny clauses merge** within one section (never across headings).
5. **Tables are separate chunks**, split by rows with the header repeated.
6. **Every chunk stores** its paragraph refs, heading path, and resolved amendments
   (action, effective date, amending circular and link) for precise citations.
7. **Contextual header.** `context` = document title › headings › para refs (+ lead-in)
   is prepended for embedding and keyword search; `text` is the verbatim regulation.
8. **Derived, versioned data.** Chunks are rebuilt from stored raw HTML; bumping
   `CHUNKER_VERSION` re-chunks everything without re-crawling.

## Consequences

- 45 documents → 3,707 chunks, median ~100 tokens, max 400; a word-level check shows
  no source text is lost.
- Citations can name exact paragraphs ("KYC Directions, para 63(9)(iv), inserted
  w.e.f. 29 Dec 2025").
- Token counts use a ~4 chars/token estimate until the embedding model is chosen;
  the counter is injectable, so the real tokenizer drops in without other changes.
- Numbering detection is heuristic; `PageLayoutChanged` and the coverage checks are
  the guard rails if RBI changes its format.
