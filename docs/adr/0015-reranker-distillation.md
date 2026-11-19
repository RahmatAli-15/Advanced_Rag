# ADR 0015: Distilling the reranker into a smaller model

**Status:** Accepted · **Date:** 2026-11-19

## Context

The cross-encoder reranker is the most accurate part of retrieval and the slowest: about
650 of 750 ms per question on the laptop GPU (bench, ADR 0006), and about **30 seconds
on CPU**. That ties the system to a GPU, which is the most expensive part of any cloud
deployment. Smaller rerankers exist, but out of the box they are clearly less accurate
on this corpus.

## Decision

**Knowledge distillation:** train a small multilingual reranker (the *student*) to
reproduce the large one (the *teacher*) on this exact task.

| | Teacher | Student |
|---|---|---|
| Model | `BAAI/bge-reranker-v2-m3` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| Parameters | 568M | 118M (4.8x smaller) |
| Licence | Apache-2.0 | Apache-2.0 |

**Training data** (`cli distill-data`, `app/distill/data.py`):
- 300 clauses sampled from the corpus, **excluding every clause used by the evaluation
  sets**, so the test measures skill, not memory.
- An LLM writes a question per clause in English, Hinglish and Hindi: 714 questions
  from 238 usable clauses.
- For each question, hybrid search returns the 30 candidates the reranker would see in
  production, and the teacher scores each: 21,420 scored pairs.
- Train/dev split **by source clause**, so the three language versions of one question
  never sit on both sides.

**Training** (`cli distill-train`, `app/distill/train.py`, a plain PyTorch loop):

```
loss = KL( softmax(teacher/T) || softmax(student/T) ) * T^2      rank like the teacher
     + BCE( sigmoid(student), teacher probability )               score like the teacher
```

Two terms because the score is used twice: to rank, and to refuse ("not in the RBI
rules") when the top score is below 0.05. Ranking alone would leave the student's scores
on another scale and break that gate. AdamW, linear warm-up and decay, fp16 autocast with
a gradient scaler, gradient clipping, accumulation over 4 questions (one question of 30
pairs per forward pass fits a 4 GB GPU), early stopping on dev loss. 3 epochs, ~48 min
each on the RTX 3050.

## Results

**Dev set (held-out clauses):**

| | Dev loss | Same best clause as teacher | Mean prob. error |
|---|---|---|---|
| Student before training | 1.53 | 80.6% | 0.231 |
| Student after training | **0.66** | **88.9%** | **0.096** |

**Retrieval test set** (144 questions, 3 languages, default hybrid config;
`eval/reports/retrieval-20261119-0225.md` and `-0307.md`):

| Reranker | hit@1 | hit@5 | MRR@10 | Retrieval p50 (GPU) |
|---|---|---|---|---|
| Teacher | 0.861 | 0.965 | 0.908 | ~730 ms |
| Student, untrained | 0.764 | 0.903 | 0.828 | 172 ms |
| **Student, distilled** | **0.833** | **0.924** | **0.879** | **172 ms** |

Distillation recovered **71% of the hit@1 gap** (9.7 → 2.8 points) and 64% of the MRR
gap; Hindi hit@5 rose from 0.875 to 0.917.

**Speed** (one rerank of 30 clauses, `eval/reports/distill-bench-20261119.md`):

| | GPU | CPU (laptop) |
|---|---|---|
| Teacher | 570 ms | 29.8 s |
| Student | **80 ms (7x)** | **3.1 s (10x)** |

**Relevance gate** (refuse when the top score < 0.05): the student refuses 0/144 real
questions (teacher 1/144) and 12/20 off-topic ones (teacher 13/20). It behaves like the
teacher, so the threshold carries over without recalibration.

A pitfall caught on the way: the base student was published with an `Identity` output,
and `save_pretrained` copied it, so the trained model returned logits, not probabilities.
Ranking was unaffected; the gate was not (a first check wrongly showed 4/144 false
refusals). `mark_sigmoid_output` now sets a sigmoid output before saving, and the check
was re-run.

## What is used where

- **GPU (default): the teacher.** It is still more accurate (hit@1 +2.8 points), and on
  a GPU its latency is acceptable. Accuracy comes first in compliance.
- **CPU or cost-limited deployments: the student**, with `RERANKER_MODEL` pointing at the
  trained model. On CPU the teacher is unusable (30 s); the student makes a CPU-only
  deployment possible, at a measured cost of 2.8 points hit@1 and 4.1 points hit@5.

## Limitations and next steps

- 3.1 s per rerank on a laptop CPU is usable but not fast. Cheaper options, in order of
  effort: rerank 15 candidates instead of 30 on CPU, export to ONNX with int8
  quantisation, shorter maximum length. Each must be re-measured on this test set.
- Training questions are LLM-generated, like the evaluation questions; the teacher is
  the ceiling: the student cannot be more right than the model it imitates.
- The trained model lives in `data/models/` (not committed); `cli distill-train`
  reproduces it from the saved training data.

## Alternatives considered

- **Use the small model without training:** measured above; 9.7 points worse at hit@1.
- **Fine-tune on hard labels (only the gold clause):** one label per question throws away
  the teacher's knowledge about the other 29 candidates; distillation uses all of it.
- **Quantise the teacher instead:** smaller gain on CPU (the model stays 568M parameters)
  and no accuracy recovery; it can be combined with the student later.
