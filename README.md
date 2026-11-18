# AdvancedRag

**A compliance copilot for Indian financial regulation, built on a production-grade RAG pipeline.**

AdvancedRag tracks circulars and master directions from Indian regulators (starting with the RBI), answers questions with exact paragraph citations, and tells compliance teams what changed and what they now have to do.

> 🚧 **Status: early development (Phase 1 — foundation).** See the [roadmap](#roadmap).

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

Design decisions are recorded in [`docs/adr/`](docs/adr).

## Quick start

Requirements: Docker, Python 3.12.

```bash
cp .env.example .env              # add your GROQ_API_KEY
docker compose up -d postgres redis

cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Then open http://localhost:8000/health/ready and http://localhost:8000/docs.

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

## Development

```bash
cd backend
ruff check . && ruff format --check .
mypy app
pytest            # integration tests use a separate advancedrag_test database
```

## Roadmap

- [x] **Phase 1 — Foundation:** FastAPI skeleton, Postgres + pgvector, Redis, CI
- [x] Polite RBI crawler, structure-preserving parser, versioned (bitemporal) storage
- [x] Clause-aware chunking with amendment tracking and full-text index
- [x] Local embeddings (bge-m3), hybrid search + RRF, cross-encoder reranking, as-of-date filtering
- [ ] **Phase 2 — Core RAG:** cited answers via the LLM gateway, chat UI
- [x] LLM gateway: fallback chain, retries honouring Retry-After, rate limiting, circuit breaker, JSON repair
- [x] Retrieval evaluation: multilingual test set, ablation of every component
- [ ] **Phase 3 — Evaluation:** human review of the test set, exact-term questions, answer faithfulness, tracing
- [ ] **Phase 4 — Compliance engine:** versioned rules ("as of date"), amendment diffs, citation graph, alerts, obligation register
- [ ] **Phase 5 — Agent:** LangGraph impact analysis, MCP server
- [ ] **Phase 6 — Production:** semantic cache, rate limiting, RBAC + multi-tenancy, cost tracking, Helm

## Disclaimer

AdvancedRag is not legal advice. Always verify against the official regulator publication.

## License

[MIT](LICENSE) © 2026 Rahmat Ali
