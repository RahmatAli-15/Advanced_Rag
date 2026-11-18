"""Load test against a running API (closed loop).

`concurrency` simulated users each send a request, wait for the reply, and send the
next, so the offered load adapts to the server: throughput rises with concurrency until
some resource saturates, after which only latency grows. Questions come from the
retrieval test set, so the load looks like real use (three languages, varied lengths).

Endpoints:
  search     retrieval + reranking: our own compute (GPU), the real capacity limit
  ask-cached repeat questions: the answer cache path (Redis + audit log write)
LLM-backed /ask is not load-tested: its throughput is set by the provider's rate limit
(requests per minute per key), not by this system, and load-testing it spends quota.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

Endpoint = Literal["search", "ask-cached"]


@dataclass
class LoadResult:
    endpoint: str
    concurrency: int
    requests: int
    wall_s: float
    latencies_ms: list[float] = field(default_factory=list)
    statuses: dict[int, int] = field(default_factory=dict)
    errors: int = 0  # connection errors and timeouts

    @property
    def ok(self) -> int:
        return self.statuses.get(200, 0)

    @property
    def throughput(self) -> float:
        return self.ok / self.wall_s if self.wall_s else 0.0

    def pct(self, p: float) -> float:
        if not self.latencies_ms:
            return 0.0
        xs = sorted(self.latencies_ms)
        return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))]


def _request(endpoint: Endpoint, question: str) -> tuple[str, dict[str, Any]]:
    if endpoint == "search":
        return "/search", {"query": question, "k": 6}
    return "/ask", {"question": question}


async def run(
    base_url: str,
    endpoint: Endpoint,
    questions: list[str],
    concurrency: int,
    requests: int,
    *,
    timeout_s: float = 120.0,
    max_tries: int = 10,
) -> LoadResult:
    result = LoadResult(endpoint, concurrency, requests, 0.0)
    next_i = 0

    async def user(client: httpx.AsyncClient) -> None:
        nonlocal next_i
        while next_i < requests:
            i, next_i = next_i, next_i + 1  # no await in between: safe in asyncio
            path, body = _request(endpoint, questions[i % len(questions)])
            # Behave like a well-written client: on 429/503, wait as long as the server
            # asks (Retry-After) and try the same request again, up to max_tries.
            for _ in range(max_tries):
                t0 = time.perf_counter()
                try:
                    r = await client.post(path, json=body)
                except httpx.HTTPError:
                    result.errors += 1
                    break
                result.statuses[r.status_code] = result.statuses.get(r.status_code, 0) + 1
                if r.status_code == 200:
                    result.latencies_ms.append((time.perf_counter() - t0) * 1000)
                if r.status_code not in (429, 503):
                    break
                await asyncio.sleep(min(float(r.headers.get("Retry-After", 1)), 10.0))

    limits = httpx.Limits(max_connections=concurrency)
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s, limits=limits) as client:
        t0 = time.perf_counter()
        await asyncio.gather(*(user(client) for _ in range(concurrency)))
        result.wall_s = time.perf_counter() - t0
    return result


def render(results: list[LoadResult]) -> str:
    lines = [
        "| Endpoint | Users | Requests | OK | Refused 503 (retried) | 429 | Errors "
        "| Throughput (req/s) | p50 ms | p95 ms | p99 ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        other = sum(n for s, n in r.statuses.items() if s not in (200, 429, 503))
        lines.append(
            f"| {r.endpoint} | {r.concurrency} | {r.requests} | {r.ok} "
            f"| {r.statuses.get(503, 0)} | {r.statuses.get(429, 0)} | {r.errors + other} "
            f"| {r.throughput:.2f} "
            f"| {r.pct(50):.0f} | {r.pct(95):.0f} | {r.pct(99):.0f} |"
        )
    return "\n".join(lines) + "\n"
