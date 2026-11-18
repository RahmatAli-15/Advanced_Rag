"""Metrics, cost accounting and the daily token budget (ADR 0011). No network: a fake
provider, an in-memory budget and a dict-backed fake Redis."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app.core import metrics
from app.llm.budget import DailyTokenBudget
from app.llm.gateway import LLMGateway, ModelRoute
from app.llm.types import (
    LLMRequest,
    LLMResponse,
    Message,
    RateLimited,
    StreamEvent,
    Usage,
)
from app.main import create_app

REQ = LLMRequest([Message("user", "hi")])


def _sample(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


class Provider:
    name = "fake"

    def __init__(self, fail_first: dict[str, Exception] | None = None) -> None:
        self.calls: list[str] = []
        self.fail_first = dict(fail_first or {})

    async def complete(self, request: LLMRequest, model: str) -> LLMResponse:
        self.calls.append(model)
        if model in self.fail_first:
            raise self.fail_first.pop(model)
        return LLMResponse("ok", model, "fake", Usage(1000, 500))

    async def stream(self, request: LLMRequest, model: str) -> Any:
        self.calls.append(model)
        yield StreamEvent(text="ok")
        yield StreamEvent(done=True, model=model, usage=Usage(1000, 500))


class Budget:
    def __init__(self, exhausted: bool) -> None:
        self.is_exhausted = exhausted
        self.added: list[int] = []

    async def exhausted(self) -> bool:
        return self.is_exhausted

    async def add(self, tokens: int) -> None:
        self.added.append(tokens)


async def _no_sleep(_: float) -> None:
    return None


PRICES = {"big": [0.15, 0.60], "small": [0.075, 0.30]}


def _gateway(p: Provider, models: list[str], budget: Budget | None = None) -> LLMGateway:
    return LLMGateway(
        [ModelRoute(p, m, requests_per_minute=6000) for m in models],
        budget=budget,
        prices=PRICES,
        sleep=_no_sleep,
    )


def test_cost_uses_list_prices_per_million_tokens() -> None:
    assert metrics.cost_usd(PRICES, "big", 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert metrics.cost_usd(PRICES, "small", 2000, 1000) == pytest.approx(0.00045)
    assert metrics.cost_usd(PRICES, "unknown", 5000, 5000) == 0.0


async def test_successful_call_records_tokens_cost_and_outcome() -> None:
    before = (
        _sample("llm_tokens_total", model="big", kind="prompt"),
        _sample("llm_cost_usd_total", model="big"),
        _sample("llm_calls_total", route="big", outcome="ok"),
    )
    await _gateway(Provider(), ["big"]).complete(REQ)
    assert _sample("llm_tokens_total", model="big", kind="prompt") - before[0] == 1000
    assert _sample("llm_cost_usd_total", model="big") - before[1] == pytest.approx(
        (1000 * 0.15 + 500 * 0.60) / 1e6
    )
    assert _sample("llm_calls_total", route="big", outcome="ok") - before[2] == 1


async def test_failed_attempts_are_counted_by_outcome() -> None:
    before = _sample("llm_calls_total", route="big", outcome="rate_limited")
    p = Provider({"big": RateLimited("429", retry_after_s=1)})
    await _gateway(p, ["big"]).complete(REQ)
    assert _sample("llm_calls_total", route="big", outcome="rate_limited") - before == 1


async def test_spent_budget_starts_at_the_cheaper_model() -> None:
    downgrades = _sample("llm_budget_downgrades_total")
    p, budget = Provider(), Budget(exhausted=True)
    gw = _gateway(p, ["big", "small"], budget)
    r = await gw.complete(REQ)
    assert r.model == "small" and p.calls == ["small"]
    assert _sample("llm_budget_downgrades_total") - downgrades == 1
    assert gw.stats.fallbacks == 0  # a planned downgrade is not a failure fallback
    assert budget.added == []  # only the primary model's tokens count against it


async def test_budget_counts_primary_model_tokens() -> None:
    p, budget = Provider(), Budget(exhausted=False)
    r = await _gateway(p, ["big", "small"], budget).complete(REQ)
    assert r.model == "big" and budget.added == [1500]


async def test_single_model_chain_ignores_the_budget() -> None:
    # Better an answer over budget than no answer: there is nothing cheaper to use.
    p = Provider()
    r = await _gateway(p, ["big"], Budget(exhausted=True)).complete(REQ)
    assert r.model == "big"


async def test_stream_respects_the_budget_and_records_usage() -> None:
    p, budget = Provider(), Budget(exhausted=True)
    before = _sample("llm_tokens_total", model="small", kind="completion")
    events = [ev async for ev in _gateway(p, ["big", "small"], budget).stream(REQ)]
    assert p.calls == ["small"] and events[-1].done
    assert _sample("llm_tokens_total", model="small", kind="completion") - before == 500


class FakeRedis:
    def __init__(self, fail: bool = False) -> None:
        self.data: dict[str, int] = {}
        self.fail = fail

    async def get(self, key: str) -> int | None:
        if self.fail:
            raise ConnectionError("redis down")
        return self.data.get(key)

    async def incrby(self, key: str, n: int) -> int:
        if self.fail:
            raise ConnectionError("redis down")
        self.data[key] = self.data.get(key, 0) + n
        return self.data[key]

    async def expire(self, key: str, seconds: int) -> bool:
        return True


async def test_daily_budget_is_per_utc_day() -> None:
    day = [date(2026, 11, 18)]
    budget = DailyTokenBudget(FakeRedis(), 1000, today=lambda: day[0])
    await budget.add(600)
    assert not await budget.exhausted()
    await budget.add(400)
    assert await budget.exhausted()
    day[0] = date(2026, 11, 19)  # a new day starts with a fresh budget
    assert not await budget.exhausted()


async def test_budget_is_ignored_when_redis_is_down() -> None:
    budget = DailyTokenBudget(FakeRedis(fail=True), 10)
    await budget.add(100)  # must not raise
    assert not await budget.exhausted()


async def test_zero_limit_turns_the_budget_off() -> None:
    redis = FakeRedis()
    budget = DailyTokenBudget(redis, 0)
    await budget.add(10**9)
    assert not await budget.exhausted()


class QueueRedis:
    async def xinfo_groups(self, stream: str) -> list[dict[str, Any]]:
        return [{"name": b"workers", "pending": 1, "lag": 2}]

    async def zcard(self, key: str) -> int:
        return 3

    async def xlen(self, key: str) -> int:
        return 4


def test_metrics_endpoint_exposes_prometheus_text() -> None:
    app = create_app()
    app.router.lifespan_context = _noop_lifespan
    app.state.redis = QueueRedis()
    with TestClient(app) as client:
        r = client.get("/metrics")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    assert 'queue_jobs{state="ready"} 2.0' in r.text
    assert 'queue_jobs{state="dead"} 4.0' in r.text
    assert "llm_calls_total" in r.text and "rag_stage_seconds" in r.text


def test_metrics_endpoint_survives_redis_errors() -> None:
    app = create_app()
    app.router.lifespan_context = _noop_lifespan
    app.state.redis = object()  # every queue call fails
    with TestClient(app) as client:
        assert client.get("/metrics").status_code == 200


@asynccontextmanager
async def _noop_lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
