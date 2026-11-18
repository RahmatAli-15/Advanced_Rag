"""Overload protection (ADR 0013): the GPU concurrency gate (load shedding) and the
per-client rate limit. Fake Redis and stub services: no Docker, no models."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import create_app
from app.retrieval.gate import ConcurrencyGate, Overloaded

# --- gate ----------------------------------------------------------------------------


async def test_gate_runs_limited_work_queues_some_and_sheds_the_rest() -> None:
    gate = ConcurrencyGate(max_concurrent=2, max_waiting=1)
    release = asyncio.Event()
    running = 0
    peak = 0

    async def work() -> None:
        nonlocal running, peak
        async with gate.slot():
            running += 1
            peak = max(peak, running)
            await release.wait()
            running -= 1

    tasks = [asyncio.create_task(work()) for _ in range(3)]  # 2 run, 1 waits
    await asyncio.sleep(0)
    with pytest.raises(Overloaded):  # the 4th is refused at once
        async with gate.slot():
            pass
    release.set()
    await asyncio.gather(*tasks)
    assert peak == 2
    async with gate.slot():  # capacity is back
        pass


async def test_gate_frees_the_slot_when_work_fails() -> None:
    gate = ConcurrencyGate(max_concurrent=1, max_waiting=0)
    with pytest.raises(ValueError):
        async with gate.slot():
            raise ValueError("boom")
    async with gate.slot():  # not leaked
        pass


async def test_gate_frees_the_waiting_place_on_cancel() -> None:
    gate = ConcurrencyGate(max_concurrent=1, max_waiting=1)
    hold = asyncio.Event()

    async def holder() -> None:
        async with gate.slot():
            await hold.wait()

    async def waiter() -> None:
        async with gate.slot():
            pass

    h = asyncio.create_task(holder())
    await asyncio.sleep(0)
    w = asyncio.create_task(waiter())
    await asyncio.sleep(0)
    w.cancel()  # e.g. the client disconnected while queued
    with pytest.raises(asyncio.CancelledError):
        await w
    w2 = asyncio.create_task(waiter())  # the waiting place is free again
    await asyncio.sleep(0)
    hold.set()
    await asyncio.gather(h, w2)


# --- API ------------------------------------------------------------------------------


class FakePipeline:
    def __init__(self, redis: "FakeRedis") -> None:
        self.redis, self.ops = redis, []

    async def __aenter__(self) -> "FakePipeline":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    def incr(self, key: str) -> None:
        self.ops.append(key)

    def expire(self, key: str, seconds: int) -> None:
        return None

    async def execute(self) -> list[Any]:
        if self.redis.fail:
            raise ConnectionError("redis down")
        key = self.ops[0]
        self.redis.counts[key] = self.redis.counts.get(key, 0) + 1
        return [self.redis.counts[key], True]


class FakeRedis:
    def __init__(self, fail: bool = False) -> None:
        self.counts: dict[str, int] = {}
        self.fail = fail

    def pipeline(self, transaction: bool = True) -> FakePipeline:
        return FakePipeline(self)


class BusyRetriever:
    async def search(self, *a: Any, **kw: Any) -> Any:
        raise Overloaded(retry_after_s=2)


class BusyAnswerService:
    async def ask(self, question: str, as_of: date | None = None) -> Any:
        raise Overloaded(retry_after_s=2)

    async def ask_stream(self, question: str, as_of: date | None = None) -> AsyncIterator[Any]:
        raise Overloaded()
        yield  # pragma: no cover  (makes this an async generator)


@asynccontextmanager
async def _noop(_: FastAPI) -> AsyncIterator[None]:
    yield


def _app(redis: Any = None, limit: int = 3) -> FastAPI:
    app = create_app()
    app.router.lifespan_context = _noop
    app.state.redis = redis
    app.state.rate_limit_per_minute = limit
    app.state.retriever = BusyRetriever()
    app.state.answer_service = BusyAnswerService()
    return app


def test_requests_over_the_limit_get_429_with_retry_after() -> None:
    with TestClient(_app(FakeRedis(), limit=3)) as client:
        codes = [client.post("/search", json={"query": "kyc"}).status_code for _ in range(5)]
        r = client.post("/ask", json={"question": "kyc?"})  # same client, same budget
    assert codes == [503, 503, 503, 429, 429]  # 503: the stub retriever is "busy"
    assert r.status_code == 429
    assert 1 <= int(r.headers["Retry-After"]) <= 60
    assert "requests per minute" in r.json()["detail"]


def test_rate_limit_fails_open_when_redis_is_down() -> None:
    with TestClient(_app(FakeRedis(fail=True), limit=1)) as client:
        codes = {client.post("/search", json={"query": "kyc"}).status_code for _ in range(3)}
    assert codes == {503}  # never 429: the limiter allowed every request


def test_zero_limit_turns_rate_limiting_off() -> None:
    with TestClient(_app(FakeRedis(), limit=0)) as client:
        codes = {client.post("/search", json={"query": "kyc"}).status_code for _ in range(5)}
    assert codes == {503}


def test_overloaded_search_and_ask_return_503_with_retry_after() -> None:
    with TestClient(_app()) as client:
        for r in (
            client.post("/search", json={"query": "kyc"}),
            client.post("/ask", json={"question": "kyc?"}),
        ):
            assert r.status_code == 503 and r.headers["Retry-After"] == "2"
            assert "busy" in r.json()["detail"]


def test_overloaded_stream_sends_a_busy_error_event() -> None:
    with TestClient(_app()) as client:
        r = client.post("/ask/stream", json={"question": "kyc?"})
    event, data = r.text.strip().split("\n")
    assert event == "event: error"
    assert "busy" in json.loads(data.removeprefix("data: "))["message"]
