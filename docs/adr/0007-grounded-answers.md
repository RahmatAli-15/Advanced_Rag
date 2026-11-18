# ADR 0007: Grounded answers with verified citations

**Status:** Accepted · **Date:** 2026-11-18

## Context

In compliance, a confident wrong answer is worse than no answer. Every statement must
trace to a clause of a regulation in force on the date asked about, and the system must
say plainly when it does not know.

## Decision

`POST /ask` runs: hybrid retrieval (as-of filtered) → gate → LLM → citation check → audit log.

1. **Gate 1, retrieval score.** If the reranker's top score is below 0.05, return
   `not_found` without calling the LLM. Calibrated on the 153-question test set plus 15
   off-topic questions: 0.7 % of in-domain questions fall below it; 12 of 15 off-topic
   ones are blocked. The 3 that pass are near-domain (SEBI mutual funds, demat accounts,
   CRR for banks), so a score threshold alone is not enough —
2. **Gate 2, the model.** The prompt requires `"answerable": false` when the sources do
   not contain the answer. The SEBI question correctly returns `insufficient_sources`.
3. **Numbered, fenced sources.** Up to 6 clauses go in `<source id="S1" document=…
   paragraphs=… amendments=…>` tags. The prompt states that instructions inside sources
   are content, not commands (prompt-injection defense).
4. **Citation verification.** Citation styles are normalised first (`gpt-oss` writes its
   native `【S1】` instead of the requested `[S1]`; others write `[S1, S3]`). Citations to
   sources we did not provide are removed and recorded. An answer that cites nothing is
   `ungrounded`, never shown as a normal answer.
5. **Honest status values:** `answered`, `not_found`, `no_coverage` (nothing in the corpus
   was in force on the as-of date — different from "irrelevant question"),
   `insufficient_sources`, `ungrounded`, `llm_unavailable` (the user still gets the
   relevant clauses).
6. **Audit log** (`answer_logs`): question, as-of date, status, answer, model, prompt
   version, retrieved and cited chunk ids, top score, tokens, timings — so any answer can
   be explained later.

## Consequences

- Observed on real questions: English, Hinglish and Hindi questions answered with exact
  paragraph citations in ~2.1–2.3 s (retrieval ~0.9 s, LLM ~1.2 s); off-topic and
  injection attempts stopped in < 1 s without an LLM call.
- The first live run showed every gpt-oss answer flagged `ungrounded` because of its
  native citation brackets. Format drift between models is now normalised and tested.
- **Not yet measured:** answer faithfulness. One answer paraphrased a table row badly
  ("₹2.5 lakh se > ₹5 lakh" for "> ₹2.5 lakh and ≤ ₹5 lakh"). The next step is an
  answer-level evaluation: claim-by-claim support checking against the cited clauses.
