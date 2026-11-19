# AdvancedRag

**A compliance copilot for Indian financial regulation, built on a production-grade RAG pipeline.**

AdvancedRag tracks circulars and master directions from Indian regulators (starting with the RBI), answers questions with exact paragraph citations, and tells compliance teams what changed and what they now have to do.

> **New here? Read [AdvancedRag, explained simply](docs/explained.md)**: what it is, why,
> how to use it, the architecture (HLD/LLD), guardrails, cost and measured results, in
> plain language.
>
> **Status:** working end to end on the RBI NBFC Master Directions (45 directions):
> cited answers in English, Hindi and Hinglish, change tracking with an explaining agent,
> monitoring and load protection. Next steps are in the [roadmap](#roadmap).

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

### Use it from Claude, Cursor or VS Code (MCP)

An MCP server gives AI assistants three read-only tools: `ask_compliance` (cited answer),
`search_regulations` (clauses verbatim) and `recent_changes`. It calls the running API,
so every answer goes through the same checks, cache and audit log
([ADR 0012](docs/adr/0012-mcp-server.md)).

Claude Code:

```bash
claude mcp add advancedrag -- /path/to/backend/.venv/bin/python -m app.mcp_server
```

Claude Desktop (`claude_desktop_config.json`; on Windows the Python path is
`backend\.venv\Scripts\python.exe`):

```json
{
  "mcpServers": {
    "advancedrag": {
      "command": "/path/to/backend/.venv/bin/python",
      "args": ["-m", "app.mcp_server"],
      "env": { "ADVANCEDRAG_API_URL": "http://localhost:8000" }
    }
  }
}
```

Then ask the assistant, e.g. *"Gold loan par LTV ki limit kitni hai?"*: it calls
`ask_compliance` and answers with the RBI direction, paragraph and link.

### Monitoring and cost control

```bash
docker compose --profile observability up -d   # Prometheus :9090, Grafana :3000
```

The API exposes Prometheus metrics at `/metrics`, and Grafana opens on a ready-made
dashboard: answers by status, cache hit rate, p95 time per stage, LLM calls by outcome
(including 429s per key), tokens and **cost per answer**, and queue depth with
dead-lettered jobs.

A **daily token budget** in the LLM gateway caps spend on the primary model: past the
limit, requests start at the cheaper model instead of failing (checked live: the next
question was answered by `gpt-oss-20b`, and the downgrade showed on the dashboard).
An uncached answer costs about $0.0004 at list price; a cache hit costs nothing
([ADR 0011](docs/adr/0011-observability-and-cost-control.md)).

### What a change means (change-impact agent)

When the worker detects a change, a small **LangGraph** agent explains it: what changed,
who it affects, what to do, and which other clauses to check. Every claim must quote the
old or new text **word for word**; code checks the quotes and sends failures back to the
model (at most twice), otherwise the paragraph is marked unverified
([ADR 0014](docs/adr/0014-change-impact-agent.md)).

Tested on a real amendment (earlier text from an Internet Archive snapshot):

```bash
python -m app.ingestion.cli changes-from-archive 12957 20260608114823
python -m app.ingestion.cli analyse-impact
```

> *Para 3(3): the exemption now applies only to NBFCs holding a Certificate of
> Registration as "Type I NBFC" (high). Para 70 lets NBFCs finance independent viable
> units as separate projects with their own financial closure (medium). Para 72
> clarifies right of way for generation projects (low).*

The analysis shows under "Recent RBI changes" in the UI and in MCP's `recent_changes`.

### A smaller reranker, trained here (knowledge distillation)

The reranker is the most accurate step and the slowest (30 s per question on CPU). A
model 4.8x smaller was trained in PyTorch to imitate it, on 21,420 teacher-scored pairs
from clauses outside the test set ([ADR 0015](docs/adr/0015-reranker-distillation.md)):

| Reranker | Right clause first | In top 5 | Rerank on GPU | Rerank on CPU |
|---|---|---|---|---|
| Teacher, 568M | 86.1% | 96.5% | 570 ms | 29.8 s |
| Small model, untrained | 76.4% | 90.3% | 80 ms | 3.1 s |
| **Small model, distilled** | **83.3%** | **92.4%** | **80 ms** | **3.1 s** |

Training recovered 71% of the accuracy gap. The teacher stays the default on a GPU; the
distilled model (`RERANKER_MODEL=../data/models/reranker-student`) makes a CPU-only
deployment possible.

```bash
python -m app.ingestion.cli distill-data     # questions + teacher scores
python -m app.ingestion.cli distill-train    # PyTorch training, ~48 min/epoch on an RTX 3050
python -m app.ingestion.cli distill-bench    # speed on GPU/CPU + relevance-gate check
```

### Capacity and overload protection

`python -m app.ingestion.cli load-test` measured the system on a laptop RTX 3050:
the GPU serves about **1.2 searches per second**, and repeat questions (answer cache)
**80-100 per second**. Without limits, 16 concurrent users collapsed search throughput to
0.5/s with a 38 s p95. Two simple protections fix that
([ADR 0013](docs/adr/0013-load-testing-and-overload-protection.md)):

- **Load shedding:** at most 2 searches use the GPU at once and 2 wait; beyond that the
  API answers 503 + Retry-After at once. With 16 users throughput stays at 1.15/s and
  p95 at **3.8 s instead of 38 s**.
- **Rate limiting:** 20 requests per minute per client (Redis), then 429 + Retry-After.

### Keep it up to date (background worker)

```bash
cd backend
python -m app.worker                      # daily RBI refresh at 06:30 IST
python -m app.worker enqueue refresh_rbi  # or run one now
python -m app.worker dead                 # inspect jobs that failed for good
```

The worker runs a job chain on a small Redis Streams queue (retries with backoff,
dead-letter queue, crash recovery, idempotent daily scheduling): crawl → chunk + embed →
paragraph diffs → impact analysis. Every detected change is written in the same transaction as the new
version (transactional outbox) and shown under "Recent RBI changes"
([ADR 0010](docs/adr/0010-background-jobs-and-change-detection.md)).

## Measured retrieval quality

144 questions in English, Hinglish and Hindi over 45 RBI NBFC Master Directions
([protocol and caveats](eval/README.md), [full analysis](docs/adr/0006-retrieval-evaluation.md)):

| Config | hit@1 | hit@5 | MRR@10 | p50 latency |
|---|---|---|---|---|
| keyword only | 0.06 | 0.23 | 0.14 | 60 ms |
| vector only | 0.72 | 0.91 | 0.80 | 117 ms |
| hybrid, equal weights | 0.44 | 0.79 | 0.58 | 118 ms |
| **hybrid (keyword ×0.3) + rerank — default** | **0.86** | **0.97** | **0.91** | 731 ms |

Equal-weight fusion made retrieval *worse* than vectors alone; the reranker is the
biggest single gain. The questions were LLM-generated, then reviewed clause by clause
(by an AI reviewer, not yet a human): 39 rewritten, 9 dropped as unanswerable from
their gold clause.

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
- [x] Prometheus + Grafana monitoring, cost per answer, daily token budget with model downgrade
- [x] MCP server: search, cited answers and recent changes as tools for AI assistants
- [x] Test set reviewed clause by clause against its gold clauses (AI review; v2)
- [x] Load test, GPU load shedding (503 + Retry-After) and per-client rate limiting
- [x] Change-impact agent (LangGraph) with quote verification, tested on a real amendment
- [x] Reranker distillation (PyTorch): 4.8x smaller, 7x faster on GPU, 71% of the accuracy gap recovered

**Next**

- [ ] Exact-term test questions (e.g. "V-CIP", "NOF") to check the keyword-search weight
- [ ] Human spot-check of the reviewed test set
- [ ] On-demand AWS deployment (Terraform)

**Deliberately not planned:** a semantic answer cache. In compliance, "Can NBFCs do
video KYC?" and "Can banks do video KYC?" are similar questions with different answers,
so only exact repeats are cached.

## Disclaimer

AdvancedRag is not legal advice. Always verify against the official regulator publication.

## License

[MIT](LICENSE) © 2026 Rahmat Ali
