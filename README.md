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
```

The crawler obeys robots.txt (RFC 9309), waits 3 s between requests, stops on any
bot-protection challenge, and is idempotent. See [ADR 0003](docs/adr/0003-crawling-policy.md).
Each Master Direction is stored with its version history, so you can ask which version
was in force on a given date.

Run the whole stack in containers instead: `docker compose --profile full up --build`.

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
- [ ] Clause-aware chunking
- [ ] **Phase 2 — Core RAG:** local embeddings, hybrid search + RRF, reranking, cited answers, chat UI
- [ ] **Phase 3 — Evaluation:** golden dataset, retrieval + faithfulness metrics gating CI, tracing
- [ ] **Phase 4 — Compliance engine:** versioned rules ("as of date"), amendment diffs, citation graph, alerts, obligation register
- [ ] **Phase 5 — Agent:** LangGraph impact analysis, MCP server
- [ ] **Phase 6 — Production:** semantic cache, rate limiting, RBAC + multi-tenancy, cost tracking, Helm

## Disclaimer

AdvancedRag is not legal advice. Always verify against the official regulator publication.

## License

[MIT](LICENSE) © 2026 Rahmat Ali
