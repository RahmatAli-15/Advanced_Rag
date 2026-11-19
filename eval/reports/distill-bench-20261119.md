# Reranker distillation: speed and relevance gate

- command: `python -m app.ingestion.cli distill-bench -n 20`
- one rerank = 1 question x 30 candidate clauses (max 512 tokens), as in production
- hardware: laptop RTX 3050 (4 GB) and the same laptop's CPU; on AC power
- generated: 2026-11-19

## Rerank latency

| Device | Model | p50 ms | p95 ms |
|---|---|---|---|
| cuda | teacher (bge-reranker-v2-m3, 568M) | 570 | 663 |
| cuda | student (distilled mMiniLMv2, 118M) | 80 | 97 |
| cpu | teacher | 29795 | 36238 |
| cpu | student | 3085 | 3650 |

## Relevance gate (refuse when the top rerank score < 0.05)

| Model | In-domain questions refused (should be 0) | Off-topic questions refused (should be 20) |
|---|---|---|
| teacher | 1 / 144 | 13 / 20 |
| student | 0 / 144 | 12 / 20 |

The student behaves like the teacher at the gate, so the 0.05 threshold needs no
recalibration. Off-topic questions that pass the gate are declined later by the answer
model (20/20 in the answer evaluation, ADR 0009).

**Correction.** A first run of this check reported the student at 4/144 and 20/20. That
run was wrong: the saved student returned raw logits instead of probabilities (its config
copied an `Identity` output from the base model), so the gate compared logits with 0.05.
Rankings, and therefore every accuracy number, were unaffected. Fixed in `train.py`
(`mark_sigmoid_output`) and re-run with `distill-bench --gate-only`.
