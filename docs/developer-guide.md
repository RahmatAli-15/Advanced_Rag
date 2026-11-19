# Developer guide

Everything needed to run, rebuild, measure and extend AdvancedRag. For what the project
is and why, start with the [README](../README.md) and [explained simply](explained.md).

Requirements: Docker, Python 3.12, Node 22. A CUDA GPU is optional (models run on CPU,
slower).

---

## 1. Run it

```bash
cp .env.example .env              # add your GROQ_API_KEY (more keys optional)
docker compose up -d postgres redis

cd backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"     # macOS/Linux: .venv/bin/python
.venv\Scripts\python.exe -m alembic upgrade head
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

Open http://localhost:8000/docs (all endpoints). Readiness: `/health/ready` (503 until the
models are loaded, ~20 s).

Chat UI (second terminal):

```bash
cd frontend
npm install          # Windows PowerShell: npm.cmd install
npm run dev          # http://localhost:5173, proxies /api to the API on :8000
```

Whole stack in containers instead: `docker compose --profile full up --build`.

> **Windows note:** if PowerShell blocks scripts (`activate`, `npm`), call
> `.venv\Scripts\python.exe` directly and use `npm.cmd`, or run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

---

## 2. Load the RBI rules

```bash
cd backend
python -m app.ingestion.cli rbi-master-directions --match "Non-Banking" --history
python -m app.ingestion.cli chunk
```

The crawler obeys robots.txt (RFC 9309), waits 3 s between requests, stops on any
bot-protection challenge, and is idempotent ([ADR 0003](adr/0003-crawling-policy.md)).
Each Master Direction is stored with its version history, so you can ask which version
was in force on a given date.

Chunking follows the regulation's own structure ([ADR 0004](adr/0004-clause-aware-chunking.md)):
one clause per chunk, exact paragraph references (`63(9)(iv)`), parent context repeated
when a clause is split, and every amendment footnote linked to the paragraph it changed.

## 3. Embed and search (local models)

```bash
# GPU (NVIDIA): install CUDA PyTorch first, then the ml extra
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[ml]"

python -m app.ingestion.cli embed          # BAAI/bge-m3, fp16 on GPU
python -m app.ingestion.cli search "can customers be verified over video" --as-of 2026-03-01
```

Or `POST /search {"query": "...", "as_of": "2026-03-01"}`. Retrieval is hybrid: pgvector
HNSW + Postgres full-text, fused with weighted RRF, reranked by `bge-reranker-v2-m3`,
restricted to the versions in force on the as-of date ([ADR 0005](adr/0005-hybrid-retrieval.md)).
Set `MODELS_BACKEND=fake` to run without downloading models.

---

## 4. Ask questions

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

- `POST /ask/stream`: Server-Sent Events (`stage`, `sources`, `delta`, `final`). Retries
  and model fallback happen only before the first token; `final` carries the verified
  answer ([ADR 0008](adr/0008-streaming-answers.md)).
- Statuses: `answered`, `not_found`, `insufficient_sources`, `no_coverage`, `ungrounded`,
  `llm_unavailable`. Every answer is written to an audit log ([ADR 0007](adr/0007-grounded-answers.md)).
- Repeat questions come from an exact-match Redis cache (~4 ms, zero tokens). The key
  includes a corpus fingerprint, so any crawl or re-embed invalidates stale answers.

## 5. Use it from Claude, Cursor or VS Code (MCP)

Three read-only tools: `ask_compliance`, `search_regulations`, `recent_changes`. The
server calls the running API ([ADR 0012](adr/0012-mcp-server.md)).

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

---

## 6. Keep it up to date (background worker)

```bash
cd backend
python -m app.worker                      # daily RBI refresh at 06:30 IST
python -m app.worker enqueue refresh_rbi  # or run one now
python -m app.worker dead                 # inspect jobs that failed for good
```

Job chain on a small Redis Streams queue (retries with backoff, dead-letter queue, crash
recovery, idempotent scheduling): crawl → chunk + embed → paragraph diffs → impact
analysis. Changes are written in the same transaction as the new version (transactional
outbox) ([ADR 0010](adr/0010-background-jobs-and-change-detection.md)).

Change-impact agent (LangGraph, [ADR 0014](adr/0014-change-impact-agent.md)):

```bash
python -m app.ingestion.cli changes-from-archive 12957 20260608114823   # real past amendment
python -m app.ingestion.cli analyse-impact
```

## 7. Monitoring and cost control

```bash
docker compose --profile observability up -d   # Prometheus :9090, Grafana :3000
```

Metrics at `/metrics`; Grafana opens on a ready-made dashboard (answers by status, cache
hit rate, p95 per stage, LLM calls by outcome, tokens, cost per answer, queue). A daily
token budget switches to the cheaper model past its limit ([ADR 0011](adr/0011-observability-and-cost-control.md)).

---

## 8. Measure

```bash
python -m app.ingestion.cli eval-retrieval        # retrieval quality, all configs, 3 languages
python -m app.ingestion.cli eval-answers -n 30    # answer faithfulness + refusals
python -m app.ingestion.cli bench-retrieval       # time per stage vs quality
python -m app.ingestion.cli load-test             # throughput and p95 of a running API
```

Reports go to [`eval/reports/`](../eval/reports). Protocols and caveats:
[`eval/README.md`](../eval/README.md), [ADR 0006](adr/0006-retrieval-evaluation.md),
[ADR 0009](adr/0009-answer-evaluation.md), [ADR 0013](adr/0013-load-testing-and-overload-protection.md).

Where the time goes (laptop RTX 3050): embed + search ~120 ms, rerank 30 clauses
~650 ms, LLM ~1,000 ms. Reranking fewer clauses is faster but less accurate (top-5:
30 → 94.1%, 15 → 90.2%, 10 → 88.2%, none → 82.4%), so it stays at 30.

## 9. Train the smaller reranker (distillation)

```bash
python -m app.ingestion.cli distill-data     # LLM questions + teacher scores (~25 min)
python -m app.ingestion.cli distill-train    # PyTorch, ~48 min/epoch on an RTX 3050
python -m app.ingestion.cli distill-bench    # GPU/CPU speed + relevance-gate check
```

Use it with `RERANKER_MODEL=../data/models/reranker-student` (relative to `backend/`).
Stop the API first: training needs the GPU memory. Keep a laptop on AC power: on
battery the GPU is throttled ([ADR 0015](adr/0015-reranker-distillation.md)).

---

## 10. Checks (same as CI)

```bash
cd backend
ruff check . && ruff format --check .
mypy app
pytest            # integration tests use a separate advancedrag_test database

cd ../frontend
npm run lint && npm test && npm run build
```

CI installs only `.[dev]` (no torch): ML code imports torch lazily or is skipped in tests.
After `alembic revision --autogenerate`, run `ruff check --fix migrations`.
