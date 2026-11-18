# ADR 0006: Measure retrieval before tuning it

**Status:** Accepted · **Date:** 2026-11-18

## Context

ADR 0005 chose hybrid retrieval (vector + keyword, RRF, rerank) on reasoning, not data.
Eyeballing a few queries suggested it worked, but one query returned a weak result and
there was no way to tell whether a change helped or hurt.

## Decision

1. **A labelled test set** (`eval/datasets/retrieval_v1.jsonl`), built by a documented,
   reproducible protocol (`eval/README.md`): 51 clauses sampled across all 45 documents
   (≤ 2 per document, seed 42); an LLM writes one question per clause in English,
   Hinglish and Hindi; questions copying ≥ 5 consecutive words from the clause are
   rejected. Labels are stable (document key + paragraph refs + text snippet), so they
   survive re-chunking. Items are `reviewed: false` until a human checks them.
2. **Metrics:** hit@1/5/10, MRR@10, doc@5 (right regulation, any clause) and latency
   p50/p95, overall and per language.
3. **Every component is ablated**: keyword, vector, hybrid, weighted hybrid, each with
   and without the reranker.

## Results (153 questions, RTX 3050)

| Config | hit@1 | hit@5 | MRR@10 | p50 |
|---|---|---|---|---|
| keyword | 0.059 | 0.209 | 0.124 | 68 ms |
| vector | 0.765 | 0.889 | 0.818 | 118 ms |
| hybrid (equal weights) | 0.490 | 0.778 | 0.607 | 118 ms |
| hybrid, keyword weight 0.3 | 0.595 | 0.824 | 0.695 | 118 ms |
| vector + rerank | 0.797 | 0.941 | 0.861 | 703 ms |
| hybrid + rerank | 0.784 | 0.928 | 0.847 | 814 ms |
| **hybrid kw 0.3 + rerank (default)** | **0.791** | **0.941** | **0.857** | **720 ms** |

Findings:
- Equal-weight RRF made results *worse* than vector search alone: keyword search is weak
  on paraphrased questions (and returns nothing for Hindi), and RRF gave its noise the
  same weight. Down-weighting it to 0.3 recovers most of the loss.
- The cross-encoder reranker is the largest single gain (+5 points hit@5 over vector).
- With reranking, vector-only and weighted hybrid are tied (differences of one question).

## Decision on defaults

Hybrid with keyword weight 0.3 plus reranking. It ties the best configuration here and
keeps exact-term matching (regulation numbers, acronyms such as "V-CIP", "NOF"), which
this paraphrase-only test set does not exercise.

## Limitations (stated, not hidden)

- **Synthetic questions.** Written by an LLM and not yet human-reviewed; spot checks found
  one question naming its document and one Hindi mistranslation.
- **Biased against keyword search** by design (no copied wording). An exact-term question
  set is the next check before the keyword weight is trusted.
- **Small:** 51 independent clauses × 3 languages. Differences below ~3 points are noise.
- **One relevant clause per question.** Rules repeated across directions (the same text in
  NBFC and HFC directions) count as misses. `doc@5` partly shows this.
- Evaluation needs the GPU models, so it runs locally, not in CI; CI runs the unit and
  integration tests with fake models.
