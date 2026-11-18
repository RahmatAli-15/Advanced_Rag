# ADR 0009: Answer evaluation (faithfulness, citations, abstention)

**Status:** Accepted · **Date:** 2026-11-18

## Context

Retrieval metrics (ADR 0006) say whether the right clause was *found*. They say nothing
about the answer the user reads: whether each sentence is supported, whether its
citation points at the clause that supports it, whether numbers are copied correctly,
and whether out-of-scope questions are declined.

## Decision

`python -m app.ingestion.cli eval-answers` runs the user-facing (streaming) pipeline and:

1. **Splits each answer into claims**, each with the citations it carries. Headings,
   table headers and fragments are not claims.
2. **Deterministic checks:** every number in a claim must appear in the clauses it
   cites; the answer must be in the question's script (Hindi → Devanagari; English and
   Hinglish → Latin).
3. **LLM judge from a different model family** (`qwen/qwen3.8-27b` judging
   `gpt-oss-120b`, to avoid self-preference). It labels every claim *supported /
   partial / unsupported / contradicted* against **all** clauses the model was given,
   **without seeing which clause the claim cited**. Comparing its "supported_by" with
   the actual citation measures citation accuracy separately from faithfulness.
4. **Judge sensitivity:** one claim per answer is planted with a wrong number. The run
   reports how many the judge caught. If the judge accepts planted errors, its
   faithfulness scores mean little.
5. **20 unanswerable questions** (other regulators, off-topic, near-domain traps) are
   scored on abstention.
6. The run uses the primary model only, with patient retries, so a fallback model or an
   outage cannot skew the scores; the report lists which model answered.

## Results (30 answerable + 20 unanswerable, seed 7)

| Metric | Baseline | After fixes |
|---|---|---|
| Answered (not falsely "ungrounded") | 0.80 | **0.97** |
| Claims supported (strict) | 0.912 | **0.959** |
| Answers with every claim supported | 0.708 | **0.862** |
| Numbers found in the cited clauses | 0.902 | **1.000** |
| Citation accuracy | 0.983 | **1.000** |
| Hinglish answers in Latin script | 2/7 | 0/9 → **9/9** (see 3 below) |
| Unanswerable questions declined | 20/20 | 20/20 |
| Judge caught planted wrong numbers | 9/9 | 10/10 |

Reports: `eval/reports/answers-20261118-1008` (baseline), `-1044` (after fixes 1–2),
`-1047` (Hinglish only, after fix 3).

## What the evaluation found, and the fixes

1. **Correct answers shown as "Unverified".** All 6 "ungrounded" baseline answers had
   citations, in formats the parser did not know: zero-width spaces inside the
   brackets, `[**S1**]`, `(S6)`, later `[स्रोत: S3]`. Fixed in backend and UI.
2. **Dropped qualifiers.** "Where possible" became "must"; "disbursed only after
   clearances" became "sanctioned only after". The prompt now requires keeping
   qualifiers and conditions verbatim. The same question now answers "where possible".
3. **Hinglish answered in Devanagari.** A general rule in the system prompt made it
   *worse* (2/7 → 0/9): the model sees Hindi and writes Devanagari. A per-question
   instruction next to the question ("Answer language: Hinglish, Latin letters"),
   chosen by a small word-list detector, fixed it (9/9).
4. **Evaluator bugs**: headings counted as claims; fixed before the "after" run.

Running it also hardened the LLM gateway: rate limits no longer open the circuit
breaker, a long Retry-After moves on instead of retrying, streams have a total
deadline, every failed attempt is logged.

## Limitations

- 30 + 20 questions: differences of a few points are within noise.
- The judge is an LLM; its sensitivity check covers wrong numbers, not every error type.
- Questions are LLM-generated and not yet human-reviewed (see eval/README.md).
- One answer still cited once at the end ("[All conditions are taken from S1]") instead
  of per sentence; it is correctly flagged "ungrounded" rather than loosening the parser.
