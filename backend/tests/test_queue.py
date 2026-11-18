"""Redis Streams job queue: delivery, ack, idempotent enqueue, retries with backoff,
dead-letter queue, crash recovery, and the daily scheduler. Needs a running Redis."""

import uuid
from collections.abc import AsyncIterator
from datetime import datetime

import pytest
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from app.core.config import get_settings
from app.queue.core import Payload, PermanentError, Queue, Worker
from app.worker import IST, schedule_daily

pytestmark = pytest.mark.integration


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
async def redis() -> AsyncIterator[Redis]:
    client = Redis.from_url(get_settings().redis_url)
    try:
        await client.ping()
    except (RedisConnectionError, OSError) as e:
        pytest.skip(f"Redis not reachable: {e}")
    yield client
    await client.aclose()


@pytest.fixture
async def queue(redis: Redis) -> AsyncIterator[Queue]:
    prefix = f"test:{uuid.uuid4().hex[:8]}:"
    q = Queue(redis, prefix=prefix, clock=Clock())
    yield q
    keys = [k async for k in redis.scan_iter(f"{prefix}*")]
    if keys:
        await redis.delete(*keys)


def _worker(q: Queue, handlers: dict, **kw: object) -> Worker:  # type: ignore[type-arg]
    return Worker(q, handlers, base_delay_s=10, **kw)  # type: ignore[arg-type]


async def _pending(w: Worker) -> int:
    info = await w.q.redis.xpending(w.q.stream, w.group)
    return int(info["pending"])


async def test_job_runs_once_and_is_acked(queue: Queue) -> None:
    seen: list[Payload] = []

    async def handler(payload: Payload, _: Queue) -> None:
        seen.append(payload)

    w = _worker(queue, {"hello": handler})
    await w.setup()
    await queue.enqueue("hello", {"x": 1})
    assert await w.run_once(block_ms=100) == 1
    assert seen == [{"x": 1}] and await _pending(w) == 0
    assert await w.run_once(block_ms=100) == 0  # nothing left


async def test_job_id_makes_enqueue_idempotent(queue: Queue) -> None:
    assert await queue.enqueue("refresh", job_id="refresh:2026-11-18")
    assert not await queue.enqueue("refresh", job_id="refresh:2026-11-18")
    assert await queue.redis.xlen(queue.stream) == 1


async def test_failed_job_is_retried_with_backoff_then_succeeds(queue: Queue) -> None:
    calls: list[int] = []

    async def flaky(_: Payload, __: Queue) -> None:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("site timed out")

    w = _worker(queue, {"flaky": flaky})
    await w.setup()
    await queue.enqueue("flaky")
    await w.run_once(block_ms=100)
    assert w.stats.retried == 1 and await queue.redis.zcard(queue.delayed) == 1

    await w.run_once(block_ms=100)  # not due yet: nothing runs
    assert len(calls) == 1

    clock: Clock = queue._clock  # type: ignore[assignment]
    clock.now += 10  # first backoff = base_delay_s
    await w.run_once(block_ms=100)
    assert len(calls) == 2 and w.stats.done == 1 and await _pending(w) == 0


async def test_job_goes_to_dead_letter_queue_after_max_attempts(queue: Queue) -> None:
    async def broken(_: Payload, __: Queue) -> None:
        raise ValueError("bad page layout")

    w = _worker(queue, {"broken": broken}, max_attempts=2)
    await w.setup()
    await queue.enqueue("broken", {"doc": "42"})
    await w.run_once(block_ms=100)
    queue._clock.now += 10  # type: ignore[attr-defined]
    await w.run_once(block_ms=100)

    dead = await queue.dead_letters()
    assert w.stats.dead == 1 and len(dead) == 1
    assert dead[0]["payload"] == {"doc": "42"} and dead[0]["attempt"] == 2
    assert "bad page layout" in dead[0]["error"]


async def test_permanent_error_is_never_retried(queue: Queue) -> None:
    async def blocked(_: Payload, __: Queue) -> None:
        raise PermanentError("bot protection: backing off")

    w = _worker(queue, {"crawl": blocked})
    await w.setup()
    await queue.enqueue("crawl")
    await w.run_once(block_ms=100)
    assert w.stats.retried == 0 and w.stats.dead == 1


async def test_job_of_a_crashed_worker_is_reclaimed(queue: Queue) -> None:
    ran: list[str] = []

    async def handler(_: Payload, __: Queue) -> None:
        ran.append("ok")

    crashed = _worker(queue, {"job": handler}, consumer="crashed")
    await crashed.setup()
    await queue.enqueue("job")
    # The crashed worker took the job but never acked it.
    await queue.redis.xreadgroup(crashed.group, "crashed", {queue.stream: ">"}, count=1)
    assert await _pending(crashed) == 1

    rescuer = _worker(queue, {"job": handler}, consumer="rescuer", visibility_timeout_s=0)
    await rescuer.run_once(block_ms=100)
    assert ran == ["ok"] and rescuer.stats.reclaimed == 1 and await _pending(rescuer) == 0


async def test_unknown_job_is_dead_lettered(queue: Queue) -> None:
    w = _worker(queue, {})
    await w.setup()
    await queue.enqueue("nope")
    await w.run_once(block_ms=100)
    assert w.stats.dead == 1


async def test_daily_schedule_runs_once_per_day_after_0630_ist(queue: Queue) -> None:
    early = datetime(2026, 11, 18, 6, 0, tzinfo=IST)
    late = datetime(2026, 11, 18, 9, 0, tzinfo=IST)
    next_day = datetime(2026, 11, 19, 7, 0, tzinfo=IST)
    assert not await schedule_daily(queue, early)
    assert await schedule_daily(queue, late)
    assert not await schedule_daily(queue, late)  # same day: already queued
    assert await schedule_daily(queue, next_day)
    assert await queue.redis.xlen(queue.stream) == 2
