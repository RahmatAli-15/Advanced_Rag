# ADR 0013: Load testing and overload protection

**Status:** Accepted · **Date:** 2026-11-18

## Context

Monitoring (ADR 0011) shows how the system behaves; it does not say how much load it
can take, or what happens beyond that. A load test answers both, and the answer decides
what protection the API needs.

## How it is measured

`python -m app.ingestion.cli load-test` (`app/eval/load.py`): a closed loop of N
simulated users, each sending a question from the retrieval test set, waiting for the
reply and sending the next. Clients behave like well-written ones: on 429/503 they wait
the server's `Retry-After` and retry (up to 10 times). Two endpoints:

- `/search`: retrieval with reranking on the local GPU, the system's own capacity limit;
- `/ask` on repeated questions: the answer-cache path.

LLM-backed answers are not load-tested: their throughput is the provider's rate limit,
not this system's, and a load test would only spend quota.

Hardware: a laptop RTX 3050 (4 GB). Runs are compared **only within one session**: the
same laptop served 1.32 searches/s for one user in the morning and 1.02 in the
afternoon. On battery, Windows throttles the GPU and results are meaningless; two runs
were discarded for that reason.

## Findings

**1. The GPU serves about 1.0-1.5 searches per second, whatever the concurrency.**
Extra users only lengthen the queue (morning baseline: p95 0.9 s with 1 user, 3.2 s
with 4, 8.8 s with 8).

**2. Without a limit, heavy load makes it collapse.** With 16 parallel searches,
throughput halved and p95 reached 38 s: parallel reranks fight over a small GPU.

**3. The answer cache is not the bottleneck:** 80-100 requests/s, p95 under 0.5 s.

## Decision

**A concurrency gate in front of the models** (`app/retrieval/gate.py`): at most 2
searches use the GPU at once and at most 2 wait; any further request is refused at once
with **503 + Retry-After: 1** instead of joining a queue (load shedding).

Same session, gate off vs on (`eval/reports/load-20261118-1517.md`, `-1520.md`):

| Users | No gate: req/s | No gate: p95 | Gate: req/s | Gate: p95 |
|---|---|---|---|---|
| 4 | 1.30 | 3.6 s | 1.16 | 3.9 s |
| 8 | 1.27 | 10.7 s | 1.16 | 3.8 s |
| 16 | **0.50** | **38 s** | 1.15 | 3.8 s |

Throughput stays flat under 4x overload and the response time of every admitted request
stays bounded. The cost: at 8-16 users some requests are refused, which is the point.
One search at a time was also tried (`load-20261118-1229.md`): equally stable, but about
20% less throughput.

**A per-client rate limit** (`app/api/ratelimit.py`) on `/ask`, `/ask/stream` and
`/search`: 20 requests per minute per client IP, counted in Redis in fixed one-minute
windows; over the limit, **429 + Retry-After**. Checked live: requests 1-20 answered,
21-25 refused with "Try again in 25 s".

Why both: the gate protects the GPU from total load; the rate limit stops one client
from taking all of it. Neither replaces the other.

**Failure behaviour:**
- Both refusals are visible on the dashboard (`retrieval_shed_total`,
  `rate_limited_total`) and say when to retry.
- The rate limit fails open if Redis is down: it protects capacity, it must not cause
  an outage.
- `/ask/stream` has already sent its 200 when retrieval starts, so a refusal there
  arrives as an `error` event with the same "busy, try again" message.
- The UI shows the server's message; the MCP server reports 503 as "retry shortly".

## Alternatives considered

- **Bigger queue instead of shedding:** everyone gets slower and clients time out
  anyway; the measured 38 s p95 is that design.
- **Sliding-window or token-bucket rate limit:** smoother at window edges, more code;
  a fixed window is enough to protect a GPU and an LLM quota.
- **Batching concurrent reranks into one GPU call:** could raise throughput, but adds a
  batching layer and its latency trade-offs. Worth it only if load requires it.

## Consequences

- Capacity of one laptop GPU is ~1.2 searches/s (~70/minute). More capacity means more
  GPU workers behind a load balancer; the gate keeps each one healthy.
- Clients are identified by IP. Behind a proxy, the proxy must pass the real client
  address (uvicorn `--proxy-headers`); `X-Forwarded-For` is not trusted directly.
- A cancelled request releases its slot at once, but a rerank already running on the
  GPU finishes in its thread.
