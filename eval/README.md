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

**Status: synthetic, not yet human-reviewed** (`"reviewed": false` on every item).
Known issues from spot checks: some questions name the document; at least one Hindi
translation is wrong ("वैकल्पिक" for "voluntary"). Review protocol: for each item, check
the question is natural, answerable from the clause alone, does not name the document,
and (for Hinglish/Hindi) is a faithful translation; fix or delete it, then set
`"reviewed": true`.

## Running

```bash
cd backend
python -m app.ingestion.cli eval-retrieval            # all configs, all languages
python -m app.ingestion.cli eval-retrieval --language hi
```

Reports are written to `reports/` as Markdown and JSON. See
[ADR 0006](../docs/adr/0006-retrieval-evaluation.md) for results and their limitations.
