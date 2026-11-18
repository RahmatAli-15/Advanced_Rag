# AdvancedRag

**A compliance copilot for Indian financial regulation, built on a production-grade RAG pipeline.**

AdvancedRag tracks circulars and master directions from Indian regulators (starting with the RBI), answers questions with exact paragraph citations, and tells compliance teams what changed and what they now have to do.

> 🚧 **Status: early development (Phase 2 — core RAG).** See the [roadmap](#roadmap).

## Why this is hard

Regulatory text breaks naive RAG:

- **Rules change over time.** Circulars amend and supersede each other, so the right answer depends on the date you ask about.
- **Amendments are instructions, not text** ("in paragraph 5(b), for 'X' substitute 'Y'").
- **Documents cite each other** by reference number, forming a graph.
- **Applicability matters.** A rule for banks may not apply to an NBFC.
- **Wrong citations are unacceptable.** Every claim must trace to a source paragraph.

## Architecture (target)

```
React + Vite UI ──▶ FastAPI ──▶ query router ──▶ hybrid search (pgvector + full-text, RRF)
                                     │                    └─▶ reranker ──▶ LLM gateway ──▶ Groq / Ollama
                                     └─▶ LangGraph impact-analysis agent
Ingestion: scheduler ─▶ crawler ─▶ queue ─▶ parse ─▶ clause-aware chunking ─▶ embed ─▶ Postgres
Events (outbox): new circular ─▶ change alerts · obligation extraction · re-index
```

The full system design (architecture, request flow, data model, patterns, failure
handling, capacity) is in [`docs/design.md`](docs/design.md); individual decisions are
recorded in [`docs/adr/`](docs/adr).

## Quick start

Requirements: Docker, Python 3.12, Node 22.

```bash
cp .env.example .env              # add your GROQ_API_KEY
docker compose up -d postgres redis

cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Then open http://localhost:8000/health/ready and http://localhost:8000/docs.

### Chat UI

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173, proxies /api to the API on :8000
```

Answers stream in token by token. Each citation opens the exact clause, the version in
force on the chosen date, its amendment history, and a link to the official RBI text.

### Ingest RBI Master Directions

```bash
cd backend
alembic upgrade head
python -m app.ingestion.cli rbi-master-directions --match "Non-Banking" --history
python -m app.ingestion.cli chunk
```

The crawler obeys robots.txt (RFC 9309), waits 3 s between requests, stops on any
bot-protection challenge, and is idempotent. See [ADR 0003](docs/adr/0003-crawling-policy.md).
Each Master Direction is stored with its version history, so you can ask which version
was in force on a given date.

### Embed and search (local models)

```bash
# GPU (NVIDIA): install CUDA PyTorch first, then the ml extra
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[ml]"

python -m app.ingestion.cli embed          # BAAI/bge-m3, fp16 on GPU
python -m app.ingestion.cli search "can customers be verified over video" --as-of 2026-03-01
```

Or via the API: `POST /search {"query": "...", "as_of": "2026-03-01"}`. Retrieval is
hybrid: pgvector HNSW + Postgres full-text, fused with RRF, reranked by
`bge-reranker-v2-m3`, restricted to the versions in force on the as-of date
([ADR 0005](docs/adr/0005-hybrid-retrieval.md)). Set `MODELS_BACKEND=fake` to run
without downloading models.

Chunking follows the regulation's own structure ([ADR 0004](docs/adr/0004-clause-aware-chunking.md)):
one clause per chunk, exact paragraph references (`63(9)(iv)`), parent context repeated
when a clause is split, and every amendment footnote linked to the paragraph it changed.

Run the whole stack in containers instead: `docker compose --profile full up --build`.

## Ask a question

```bash
curl -X POST localhost:8000/ask -H "content-type: application/json" \
  -d '{"question": "How often must an NBFC update KYC for high-risk customers?"}'
```

```json
{
  "status": "answered",
  "answer": "An NBFC must update the KYC of high-risk customers **at least once every two years** from the date of opening of the account or the last KYC updation [S1]",
  "citations": [{"id": "S1", "document_title": "Reserve Bank of India (Non-Banking Financial Companies – Know Your Customer) Directions, 2025",
                 "para_refs": ["42(1)", "42(2)", "42(3)"], "version_valid_from": "2025-12-29", "...": "..."}],
  "timings_ms": {"retrieval": 905, "llm": 1324, "total": 2229}
}
```

For a streamed answer, use `POST /ask/stream` (Server-Sent Events: `stage`, `sources`,
`delta`, `final`). Retries and model fallback happen only before the first token, and the
`final` event carries the verified answer ([ADR 0008](docs/adr/0008-streaming-answers.md)).

Works in English, Hinglish and Hindi. Unanswerable or off-topic questions get an honest
`not_found` / `insufficient_sources` / `no_coverage` instead of a guess; every citation is
verified against the clauses actually retrieved, and every answer is written to an audit
log ([ADR 0007](docs/adr/0007-grounded-answers.md)).

### Caching

Repeat questions are answered from an exact-match Redis cache: **~2.5 s → ~4 ms, zero
LLM tokens**. The key is the normalised question + as-of date + prompt version + a
corpus fingerprint, so any crawl or re-embed invalidates stale answers automatically.
Only settled outcomes are cached; cache hits are still written to the audit log and
labelled in the UI. There is deliberately no "semantic" cache: in compliance, "Can NBFCs
do video KYC?" and "Can banks do video KYC?" are similar questions with different
answers.

### Where the time goes

Measured on 153 questions (laptop RTX 3050), `python -m app.ingestion.cli bench-retrieval`:

| Step | Median |
|---|---|
| Embed the question + vector and keyword search | ~120 ms |
| Rerank 30 candidate clauses (cross-encoder) | ~650 ms |
| LLM writes the answer | ~1,000 ms |

Reranking fewer clauses is faster but less accurate (top-5 hit rate: 30 clauses 94.1%,
15 → 90.2%, 10 → 88.2%, none → 82.4%), so the reranker stays at 30: in compliance a
correct answer is worth half a second. Latency is handled instead by the cache (repeat
questions ~4 ms) and by streaming (sources after ~1 s, then the answer as it is written).
Full table: [`eval/reports/bench-20261118-1059.md`](eval/reports/bench-20261118-1059.md).

### Keep it up to date (background worker)

```bash
cd backend
python -m app.worker                      # daily RBI refresh at 06:30 IST
python -m app.worker enqueue refresh_rbi  # or run one now
python -m app.worker dead                 # inspect jobs that failed for good
```

The worker runs a job chain on a small Redis Streams queue (retries with backoff,
dead-letter queue, crash recovery, idempotent daily scheduling): crawl → chunk + embed →
paragraph diffs. Every detected change is written in the same transaction as the new
version (transactional outbox) and shown under "Recent RBI changes"
([ADR 0010](docs/adr/0010-background-jobs-and-change-detection.md)).

## Measured retrieval quality

153 questions in English, Hinglish and Hindi over 45 RBI NBFC Master Directions
([protocol and caveats](eval/README.md), [full analysis](docs/adr/0006-retrieval-evaluation.md)):

| Config | hit@1 | hit@5 | MRR@10 | p50 latency |
|---|---|---|---|---|
| keyword only | 0.06 | 0.21 | 0.12 | 68 ms |
| vector only | 0.77 | 0.89 | 0.82 | 118 ms |
| hybrid, equal weights | 0.49 | 0.78 | 0.61 | 118 ms |
| **hybrid (keyword ×0.3) + rerank — default** | **0.79** | **0.94** | **0.86** | 720 ms |

Equal-weight fusion made retrieval *worse* than vectors alone; the reranker is the
biggest single gain. The questions are LLM-generated and not yet human-reviewed.

## Measured answer quality

30 answerable questions (judged claim by claim by a different model family, blind to the
citations) and 20 questions the system should decline
([ADR 0009](docs/adr/0009-answer-evaluation.md)):

| Metric | Baseline | After the fixes it found |
|---|---|---|
| Claims supported by their sources | 91.2% | **95.9%** |
| Answers with every claim supported | 70.8% | **86.2%** |
| Numbers found in the cited clause | 90.2% | **100%** |
| Correct answers wrongly flagged "unverified" | 6 / 30 | **1 / 30** |
| Hinglish answers in the question's script | 2 / 7 | **9 / 9** |
| Out-of-scope questions declined | 20 / 20 | 20 / 20 |
| Judge caught planted wrong numbers | 9 / 9 | 10 / 10 |

```bash
python -m app.ingestion.cli eval-answers -n 30   # or --language hinglish for a quick check
```

## Development

```bash
cd backend
ruff check . && ruff format --check .
mypy app
pytest            # integration tests use a separate advancedrag_test database

cd ../frontend
npm run lint && npm test && npm run build
```

## Roadmap

- [x] **Phase 1 — Foundation:** FastAPI skeleton, Postgres + pgvector, Redis, CI
- [x] Polite RBI crawler, structure-preserving parser, versioned (bitemporal) storage
- [x] Clause-aware chunking with amendment tracking and full-text index
- [x] Local embeddings (bge-m3), hybrid search + RRF, cross-encoder reranking, as-of-date filtering
- [x] Cited answers (`/ask`): two-layer abstention, citation verification, prompt-injection fencing, audit log
- [x] **Phase 2 — Core RAG:** chat UI (React + Vite), SSE streaming, clause viewer, as-of date picker
- [x] LLM gateway: fallback chain, retries honouring Retry-After, rate limiting, circuit breaker, JSON repair
- [x] Retrieval evaluation: multilingual test set, ablation of every component
- [x] Answer evaluation: claim-level faithfulness, blind citation check, judge sensitivity, abstention
- [x] Answer cache, multi-key LLM routing, background worker (queue, retries, DLQ, daily crawl), change detection
- [ ] **Phase 3 — Evaluation:** human review of the test set, exact-term questions, tracing
- [ ] **Phase 4 — Compliance engine:** versioned rules ("as of date"), amendment diffs, citation graph, alerts, obligation register
- [ ] **Phase 5 — Agent:** LangGraph impact analysis, MCP server
- [ ] **Phase 6 — Production:** semantic cache, rate limiting, RBAC + multi-tenancy, cost tracking, Helm

## Disclaimer

AdvancedRag is not legal advice. Always verify against the official regulator publication.

## License

[MIT](LICENSE) © 2026 Rahmat Ali
