# ADR 0001: Postgres + pgvector as the single data store

**Status:** Accepted · **Date:** 2026-11-18

## Context

We store regulatory documents, their versions, chunks, embeddings, a citation graph
between documents, and (later) per-tenant data. Queries often combine all of these:
"chunks similar to X, from documents in force on date D, that apply to NBFCs".

## Decision

Use PostgreSQL with the pgvector extension for relational data, vectors and
full-text search. No separate vector database.

## Consequences

- **+** One transaction covers a document version and its chunks, so the index can
  never point at a half-ingested document.
- **+** Version, applicability and tenant filters are plain SQL `WHERE` clauses joined
  with vector search, and Row-Level Security can enforce tenant isolation.
- **+** Hybrid search (pgvector + `tsvector`) without syncing two systems.
- **+** One fewer service to run, back up and monitor.
- **−** At tens of millions of vectors, a dedicated engine (Qdrant, Milvus) may serve
  ANN queries faster. Retrieval sits behind a `Retriever` interface so it can be
  swapped if measurements show the need.
