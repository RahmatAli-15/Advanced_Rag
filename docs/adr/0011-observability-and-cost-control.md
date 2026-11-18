# ADR 0011: Observability and cost control

**Status:** Accepted · **Date:** 2026-11-18

## Context

The evaluations (ADR 0006, 0009) measure quality offline. In operation, three questions
need an answer at any moment, without reading logs:

1. **Is it fast?** Which stage is slow: retrieval, reranking or the LLM?
2. **Is it working?** How many answers are declined, ungrounded or failing; is the LLM
   provider rate-limiting us; are background jobs dying?
3. **What does it cost?** Tokens and money per answer, and how much the cache saves.

LLM cost also needs a hard limit: one bad day of traffic (or a loop in a client) should
not spend the month's budget.

## Decision

**Prometheus metrics, recorded at three choke points** (`app/core/metrics.py`):

| Where | Metrics |
|---|---|
| `AnswerService._log` (every answer shown, cache hits included) | `rag_answers_total{mode,status,cached}`, `rag_stage_seconds{stage,cached}` (histogram) |
| `LLMGateway` (every attempt) | `llm_calls_total{route,outcome}`, `llm_tokens_total{model,kind}`, `llm_cost_usd_total{model}`, `llm_fallbacks_total` |
| `/metrics` handler (read from Redis at scrape time) | `queue_jobs{state}`: ready, running, delayed, dead |

Recording at choke points, not throughout the code, means the numbers cannot disagree
with the audit log, and a new code path is covered automatically. Labels come only from
small fixed sets (status, stage, model, route), never from questions or users.

Cost is computed at the provider's **list price** (`LLM_PRICES_USD_PER_M`), even on a
free tier: the dashboard shows what this traffic would cost in production.

**A daily token budget in the gateway** (`app/llm/budget.py`). Tokens used by the
primary model are counted in Redis per UTC day (shared by every API process). Once the
count passes `LLM_DAILY_TOKEN_BUDGET`, requests **start at the next, cheaper model**
instead of failing: graceful degradation, not an outage. Each downgrade is logged and
counted (`llm_budget_downgrades_total`). Deliberate limits:

- With only one model configured the budget is ignored: an answer over budget is better
  than no answer.
- If Redis is down the budget is ignored (it controls cost, it must never block answers).
- A downgrade is not counted as a fallback, so the failure metric stays meaningful.

**Prometheus + Grafana in docker-compose** (`--profile observability`), with the data
source and the dashboard provisioned from files in `ops/`: one command, no clicking.

## Results (checked live, 2026-11-18)

- A cache hit was served in 5 ms with 0 tokens, against 2,960 ms and ~1,900 tokens for
  the same question uncached.
- With the budget counter set past the limit, the next question was answered by
  `gpt-oss-20b` instead of `gpt-oss-120b`, and the downgrade appeared in the metrics.
- Uncached answers cost about $0.0004 each at list price on `gpt-oss-120b`.

## Alternatives considered

- **OpenTelemetry tracing**: per-request traces are more detailed, but need a collector
  and a trace backend. Per-stage histograms answer "which stage is slow" at a fraction
  of the setup. Tracing is the next step if per-request debugging is needed.
- **A spend limit at the provider**: coarse (the whole account stops) and not available
  on every tier. The gateway budget degrades instead of stopping.
- **A metrics endpoint on the worker**: the queue's state is already in Redis, so the
  API reads it at scrape time; one scrape target instead of two.

## Consequences

- Histograms are per process. With several API replicas Prometheus scrapes each and the
  dashboard sums them (`sum by (le)`), which is correct for histograms.
- The budget is approximate under concurrency: requests already in flight when the
  limit is reached still finish on the primary model.
- Grafana runs with anonymous read-only access. That is fine locally and must be
  replaced with real authentication before any public deployment.
