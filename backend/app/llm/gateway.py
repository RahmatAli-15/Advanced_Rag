"""LLM gateway: the only way the application calls a language model (ADR 0002).

For each request it walks a fallback chain of models. Per model it applies:
  - a client-side rate limiter (token bucket), so we stay inside provider limits
    instead of discovering them through 429s
  - retries with exponential backoff + jitter, honouring Retry-After
  - a circuit breaker: after N consecutive failures the model is skipped for a
    cool-down period, so a dead model does not add latency to every request
Fatal errors (bad request, auth) are not retried. When every model fails, the caller
gets AllModelsFailed and can degrade gracefully (e.g. return search results only).

Streaming uses the same policy, but only until the first token reaches the caller.
After that a failure raises StreamInterrupted instead of retrying or falling back.
"""

import asyncio
import logging
import random
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from app.llm.types import (
    AllModelsFailed,
    FatalError,
    LLMError,
    LLMRequest,
    LLMResponse,
    Message,
    RateLimited,
    StreamEvent,
    StreamInterrupted,
    TransientError,
    Usage,
)

log = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], float]


class Provider(Protocol):
    name: str

    async def complete(self, request: LLMRequest, model: str) -> LLMResponse: ...

    def stream(self, request: LLMRequest, model: str) -> AsyncGenerator[StreamEvent]: ...


class TokenBucket:
    """`rate` requests per second on average, with bursts up to `capacity`."""

    def __init__(self, rate: float, capacity: float, clock: Clock, sleep: Sleep) -> None:
        self.rate, self.capacity = rate, capacity
        self._tokens = capacity
        self._clock, self._sleep = clock, sleep
        self._last = clock()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = self._clock()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                await self._sleep((1 - self._tokens) / self.rate)


class CircuitBreaker:
    """closed -> (N consecutive failures) -> open -> (cool-down) -> half-open -> one trial."""

    def __init__(self, failure_threshold: int, cooldown_s: float, clock: Clock) -> None:
        self.failure_threshold, self.cooldown_s = failure_threshold, cooldown_s
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        return "half_open" if self._clock() - self._opened_at >= self.cooldown_s else "open"

    def allow(self) -> bool:
        return self.state != "open"

    def record_success(self) -> None:
        self._failures, self._opened_at = 0, None

    def record_failure(self) -> None:
        self._failures += 1
        if self.state == "half_open" or self._failures >= self.failure_threshold:
            self._opened_at = self._clock()


@dataclass(frozen=True)
class ModelRoute:
    provider: Provider
    model: str
    requests_per_minute: float = 30.0


@dataclass
class GatewayStats:
    calls: int = 0
    retries: int = 0
    fallbacks: int = 0
    failures: int = 0
    usage_by_model: dict[str, Usage] = field(default_factory=dict)


class LLMGateway:
    def __init__(
        self,
        routes: list[ModelRoute],
        *,
        max_retries: int = 3,
        breaker_threshold: int = 3,
        breaker_cooldown_s: float = 60.0,
        max_backoff_s: float = 30.0,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if not routes:
            raise ValueError("at least one model route is required")
        self.routes = routes
        self.max_retries = max_retries
        self.max_backoff_s = max_backoff_s
        self._sleep = sleep
        self._buckets = {
            r.model: TokenBucket(
                r.requests_per_minute / 60, max(1.0, r.requests_per_minute / 10), clock, sleep
            )
            for r in routes
        }
        self._breakers = {
            r.model: CircuitBreaker(breaker_threshold, breaker_cooldown_s, clock) for r in routes
        }
        self.stats = GatewayStats()

    def breaker_state(self, model: str) -> str:
        return self._breakers[model].state

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, self.max_backoff_s)
        return float(min(self.max_backoff_s, 2**attempt) + random.uniform(0, 0.5))  # noqa: S311

    async def complete(self, request: LLMRequest) -> LLMResponse:
        attempts = 0
        errors: list[str] = []
        for i, route in enumerate(self.routes):
            breaker = self._breakers[route.model]
            if not breaker.allow():
                errors.append(f"{route.model}: circuit open")
                continue
            if i > 0:
                self.stats.fallbacks += 1
                log.warning("falling back to %s", route.model)
            for attempt in range(self.max_retries + 1):
                await self._buckets[route.model].acquire()
                attempts += 1
                self.stats.calls += 1
                try:
                    response = await route.provider.complete(request, route.model)
                except FatalError as e:
                    errors.append(f"{route.model}: {e}")
                    breaker.record_failure()
                    break  # same request will fail again; try the next model
                except (RateLimited, TransientError) as e:
                    breaker.record_failure()
                    errors.append(f"{route.model}: {e}")
                    if attempt == self.max_retries or not breaker.allow():
                        break
                    self.stats.retries += 1
                    retry_after = e.retry_after_s if isinstance(e, RateLimited) else None
                    await self._sleep(self._backoff(attempt, retry_after))
                    continue
                breaker.record_success()
                prev = self.stats.usage_by_model.get(route.model, Usage())
                self.stats.usage_by_model[route.model] = Usage(
                    prev.prompt_tokens + response.usage.prompt_tokens,
                    prev.completion_tokens + response.usage.completion_tokens,
                )
                return LLMResponse(
                    response.text,
                    response.model,
                    response.provider,
                    response.usage,
                    response.latency_ms,
                    attempts,
                )
        self.stats.failures += 1
        raise AllModelsFailed("; ".join(errors[-6:]) or "no model available")

    def _record_usage(self, model: str, usage: Usage) -> None:
        prev = self.stats.usage_by_model.get(model, Usage())
        self.stats.usage_by_model[model] = Usage(
            prev.prompt_tokens + usage.prompt_tokens,
            prev.completion_tokens + usage.completion_tokens,
        )

    async def stream(self, request: LLMRequest) -> AsyncGenerator[StreamEvent]:
        """Yield text deltas, then one done=True event. Retries and fallbacks happen
        only until the first event is out; the caller never sees two models' text."""
        errors: list[str] = []
        for i, route in enumerate(self.routes):
            breaker = self._breakers[route.model]
            if not breaker.allow():
                errors.append(f"{route.model}: circuit open")
                continue
            if i > 0:
                self.stats.fallbacks += 1
                log.warning("falling back to %s (stream)", route.model)
            for attempt in range(self.max_retries + 1):
                await self._buckets[route.model].acquire()
                self.stats.calls += 1
                events = route.provider.stream(request, route.model)
                try:
                    first = await anext(events)
                except StopAsyncIteration:
                    first = StreamEvent(done=True, model=route.model, usage=Usage())
                except FatalError as e:
                    errors.append(f"{route.model}: {e}")
                    breaker.record_failure()
                    break
                except (RateLimited, TransientError) as e:
                    breaker.record_failure()
                    errors.append(f"{route.model}: {e}")
                    if attempt == self.max_retries or not breaker.allow():
                        break
                    self.stats.retries += 1
                    retry_after = e.retry_after_s if isinstance(e, RateLimited) else None
                    await self._sleep(self._backoff(attempt, retry_after))
                    continue

                breaker.record_success()
                try:
                    event: StreamEvent | None = first
                    while event is not None:
                        if event.done:
                            self._record_usage(route.model, event.usage or Usage())
                        yield event
                        if event.done:
                            return
                        event = await anext(events, None)
                except LLMError as e:
                    breaker.record_failure()
                    self.stats.failures += 1
                    raise StreamInterrupted(f"{route.model}: {e}") from e
                finally:
                    await events.aclose()  # releases the HTTP connection on cancel too
                return
        self.stats.failures += 1
        raise AllModelsFailed("; ".join(errors[-6:]) or "no model available")


async def complete_json_with_response[T](
    gateway: LLMGateway,
    request: LLMRequest,
    parse: Callable[[str], T],
    *,
    repairs: int = 1,
) -> tuple[T, LLMResponse]:
    """Ask for JSON, validate it, and on failure send the error back once for repair.
    Returns the parsed value and the response that produced it (model, usage)."""
    req = LLMRequest(request.messages, request.temperature, request.max_tokens, json_mode=True)
    last_error = ""
    for _ in range(repairs + 1):
        response = await gateway.complete(req)
        try:
            return parse(response.text), response
        except (ValueError, LLMError) as e:  # pydantic ValidationError is a ValueError
            last_error = str(e)[:500]
            req = LLMRequest(
                [
                    *request.messages,
                    Message("assistant", response.text[:4000]),
                    Message(
                        "user",
                        f"That output was invalid: {last_error}\n"
                        "Reply again with only the corrected JSON.",
                    ),
                ],
                request.temperature,
                request.max_tokens,
                json_mode=True,
            )
    raise ValueError(f"model did not produce valid JSON after repair: {last_error}")


async def complete_json[T](
    gateway: LLMGateway,
    request: LLMRequest,
    parse: Callable[[str], T],
    *,
    repairs: int = 1,
) -> T:
    value, _ = await complete_json_with_response(gateway, request, parse, repairs=repairs)
    return value
