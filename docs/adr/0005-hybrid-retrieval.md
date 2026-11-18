# ADR 0005: Hybrid retrieval with local models

**Status:** Accepted · **Date:** 2026-11-18

## Context

Regulatory questions mix exact terms ("V-CIP", "Form A", "45-IA") with paraphrase
("can a customer be verified over video?"). Keyword search misses paraphrase; vector
search blurs exact identifiers and numbers. Users may ask in Hindi or Hinglish while the
documents are English. Answers must come only from rules in force on the date asked
about. Hardware: a 4 GB laptop GPU; data is public, but local models avoid per-call
cost and keep the path open for on-premise deployment.

## Decision

- **Embeddings: BAAI/bge-m3** (1024-d, multilingual, 8K context), fp16 on GPU.
  We embed `context + text`, so the vector knows the document, headings and parent clause.
- **Keyword search: Postgres full-text** over the same `context + text`, with an OR
  query (`websearch_to_tsquery` ANDs every word, which fails on natural questions)
  ranked by `ts_rank_cd`.
- **Fusion: Reciprocal Rank Fusion (k = 60)** of the top 40 from each. Ranks, not
  scores, because cosine similarity and `ts_rank` are on unrelated scales.
- **Reranking: BAAI/bge-reranker-v2-m3** cross-encoder over the top 30 fused.
- **Time filter in both searches:** only chunks of versions with captured text whose
  `[valid_from, valid_to)` contains the as-of date (default today).
- **HNSW index (cosine), `hnsw.iterative_scan = relaxed_order`:** with a selective
  filter, a plain HNSW scan returns its `ef_search` nearest rows and *then* filters,
  silently returning fewer results. Iterative scan keeps going until enough rows pass.
- **Vectors in their own table keyed by (chunk, model)**, so a model change is a
  backfill next to the old vectors and a config switch — no downtime, easy rollback.
- **Models behind `Embedder` / `Reranker` protocols**, loaded lazily; inference runs in
  a worker thread so the event loop is never blocked. Tests use deterministic fakes, and
  the ML stack is an optional `[ml]` extra, so CI stays fast.

## Consequences

- Keyword search runs while the query is embedded, hiding its latency.
- Each search returns a trace (per-stage timings, candidate counts, every hit's vector
  rank, keyword rank, fused and rerank scores) for debugging and evaluation.
- Quality claims are not yet measured. The next step builds a golden question set and
  reports recall@k / MRR for vector-only, keyword-only, hybrid and hybrid + rerank.
- If a version in force on the as-of date was never captured (RBI lists only its date),
  that document contributes nothing for that date. This is reported, not papered over.
