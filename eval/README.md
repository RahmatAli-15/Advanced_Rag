# Evaluation

## Retrieval test set: `datasets/retrieval_v1.jsonl`

153 questions (51 clauses × English, Hinglish, Hindi) over the 45 RBI NBFC Master
Directions in force on 2026-11-18.

**How it was made** (`python -m app.ingestion.cli eval-generate -n 60`)

1. Sample text chunks of 60–400 tokens with paragraph references from versions in force,
   at most 2 per document, random seed 42.
2. `openai/gpt-oss-120b` (via the LLM gateway) writes the question a compliance officer
   would ask for each clause, in three languages, or marks the clause unusable.
3. Questions copying 5+ consecutive words from the clause are rejected (9 of 60 were).
4. Each item records the document key, paragraph refs and the first 120 characters of
   the clause, so labels stay valid if the chunker changes.

**Status: synthetic; superseded by v2 below.** Kept unchanged so earlier reports stay
reproducible.

## Reviewed test set: `datasets/retrieval_v2.jsonl` (default)

144 questions: v1 after a clause-by-clause review. **AI-reviewed, not human-reviewed**:
every item has `"reviewed_by": "ai:claude"` and `"reviewed": false` (that flag is
reserved for a person). All decisions and their reasons are in
[`datasets/review_v1.json`](datasets/review_v1.json); rewritten items keep their
`original_question`.

Each question was read against the full text of its gold clause and checked for:

1. **Answerable** from the gold clause alone
2. **Clear** about what is asked
3. **Fair**: does not copy the clause or list the answer in the question
4. **Natural, correct Hinglish/Hindi**
5. **Right label**: the gold clause really is *the* answer

| Outcome | Clauses | Questions |
|---|---|---|
| OK as written | 32 | 96 |
| Fixed | 16 | 48 (39 rewritten) |
| Rejected: the gold clause does not contain the answer | 3 | 9 |

What the fixes were: 4 factual or terminology errors (CI ≠ CIC; DLG is a *default loss*
guarantee; LWD is the *List* of Wilful Defaulters; "promoter" vs "promoter group"),
4 questions that copied the clause or gave the answer away, 4 ambiguous ones, 3 Hindi
language errors, 1 vague one.

One deliberate change of rule: v1 tried to keep document names out of questions. Where
a clause's text is shared almost word for word by many directions (repeal-and-saving,
"undefined terms", applicability lists), a question without the direction's name has no
single right answer, and real users do say "under the KYC Directions". Those questions
now name the direction.

A human pass is still the next step: spot-check the `"review": "ok"` items and confirm
the fixes, then set `"reviewed": true` on the items checked.

## Unanswerable set: `datasets/unanswerable_v1.jsonl`

20 questions the system should decline (other regulators, off-topic, near-domain traps),
used by `eval-answers`. Checked against the corpus scope (AI review): none is answered
by the RBI NBFC directions.

## Running

```bash
cd backend
python -m app.ingestion.cli eval-retrieval            # all configs, all languages (v2)
python -m app.ingestion.cli eval-retrieval --language hi
python -m app.ingestion.cli eval-answers -n 30        # answer faithfulness (ADR 0009)
python -m app.ingestion.cli bench-retrieval           # latency per stage vs quality
```

Reports are written to `reports/` as Markdown and JSON. See
[ADR 0006](../docs/adr/0006-retrieval-evaluation.md) for results and their limitations.
