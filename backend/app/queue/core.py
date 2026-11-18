"""A small, reliable job queue on Redis Streams.

    enqueue ─XADD─▶ stream "q:jobs" ─XREADGROUP─▶ worker ─ok─▶ XACK (done)
                                                   │
                                                 error
                                                   ├─ retries left ─▶ "q:delayed"
                                                   │   (sorted by due time, then back
                                                   │    into the stream)
                                                   └─ none left ────▶ "q:dead" (DLQ)

- Consumer group: several workers can share one stream; each job goes to one worker.
- At-least-once delivery: a job leaves the stream's pending list only on XACK. If a
  worker dies mid-job, another worker reclaims it after `visibility_timeout_s`
  (XAUTOCLAIM). Handlers must therefore be idempotent; ours are (crawl/chunk/embed
  skip work already done).
- Idempotent enqueue: a job_id (e.g. "refresh-rbi:2026-11-18") is claimed with
  SET NX, so the same job cannot be queued twice, even by several schedulers.
- Retries back off exponentially (30 s, 60 s, ...); after `max_attempts` the job is
  parked in the dead-letter stream with its error, for a human to inspect.

Written by hand rather than using a library: arq and saq pin redis-py below the
version this project uses, and the whole thing is small enough to read in one go.
"""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import ResponseError

log = logging.getLogger(__name__)

Payload = dict[str, Any]
Clock = Callable[[], float]


def _s(v: bytes | str) -> str:
    return v.decode() if isinstance(v, bytes) else v


class Queue:
    def __init__(self, redis: Redis, prefix: str = "q:", clock: Clock = time.time) -> None:
        self.redis = redis
        self.stream = f"{prefix}jobs"
        self.delayed = f"{prefix}delayed"
        self.dead = f"{prefix}dead"
        self._ids = f"{prefix}id:"
        self._clock = clock

    async def enqueue(
        self,
        name: str,
        payload: Payload | None = None,
        *,
        job_id: str | None = None,
        delay_s: float = 0,
        id_ttl_s: int = 7 * 86400,
    ) -> bool:
        """Queue a job. Returns False if a job with this job_id was already queued."""
        if job_id and not await self.redis.set(self._ids + job_id, "1", nx=True, ex=id_ttl_s):
            return False
        await self._push({"name": name, "payload": payload or {}, "attempt": 1}, delay_s)
        return True

    async def _push(self, job: Payload, delay_s: float = 0) -> None:
        fields: dict[Any, Any] = {"job": json.dumps(job)}
        if delay_s > 0:
            await self.redis.zadd(self.delayed, {fields["job"]: self._clock() + delay_s})
        else:
            await self.redis.xadd(self.stream, fields, maxlen=10_000, approximate=True)

    async def promote_due(self) -> int:
        """Move delayed jobs whose time has come into the stream. ZREM is the claim, so
        with several workers each delayed job is promoted exactly once."""
        due: list[bytes] = await self.redis.zrangebyscore(self.delayed, "-inf", self._clock())  # type: ignore[assignment]
        moved = 0
        for raw in due:
            if await self.redis.zrem(self.delayed, raw):
                entry: dict[Any, Any] = {"job": _s(raw)}
                await self.redis.xadd(self.stream, entry, maxlen=10_000, approximate=True)
                moved += 1
        return moved

    async def depth(self, group: str = "workers") -> dict[str, int]:
        """Jobs by state, for monitoring: waiting for a worker (ready), taken but not
        acked (running), scheduled for a retry (delayed), and dead-lettered (dead)."""
        ready = running = 0
        try:
            groups: Any = await self.redis.xinfo_groups(self.stream)
        except ResponseError:  # the stream does not exist yet: nothing enqueued
            groups = []
        for g in groups:
            if _s(g["name"]) == group:
                ready, running = int(g.get("lag") or 0), int(g["pending"])
        return {
            "ready": ready,
            "running": running,
            "delayed": int(await self.redis.zcard(self.delayed)),
            "dead": int(await self.redis.xlen(self.dead)),
        }

    async def dead_letters(self, count: int = 50) -> list[Payload]:
        rows: Any = await self.redis.xrevrange(self.dead, count=count)
        return [json.loads(_s(fields.get(b"job") or fields["job"])) for _, fields in rows]


Handler = Callable[[Payload, Queue], Awaitable[None]]


class PermanentError(Exception):
    """Raise from a handler when retrying cannot help (or must not happen, e.g. a site
    asked us to back off). The job goes straight to the dead-letter queue."""


@dataclass
class WorkerStats:
    done: int = 0
    retried: int = 0
    dead: int = 0
    reclaimed: int = 0


class Worker:
    def __init__(
        self,
        queue: Queue,
        handlers: dict[str, Handler],
        *,
        consumer: str = "worker-1",
        group: str = "workers",
        max_attempts: int = 3,
        base_delay_s: float = 30.0,
        visibility_timeout_s: float = 1800.0,
    ) -> None:
        self.q, self.handlers = queue, handlers
        self.consumer, self.group = consumer, group
        self.max_attempts, self.base_delay_s = max_attempts, base_delay_s
        self.visibility_ms = int(visibility_timeout_s * 1000)
        self.stats = WorkerStats()

    async def setup(self) -> None:
        try:
            await self.q.redis.xgroup_create(self.q.stream, self.group, id="0", mkstream=True)
        except ResponseError as e:
            if "BUSYGROUP" not in str(e):  # the group already exists: fine
                raise

    async def run_once(self, block_ms: int = 5000) -> int:
        """Promote due retries, reclaim jobs from dead workers, then run one new job.
        Returns the number of jobs handled."""
        await self.q.promote_due()
        handled = 0
        claim: Any = await self.q.redis.xautoclaim(
            self.q.stream, self.group, self.consumer, self.visibility_ms, "0-0", count=10
        )
        claimed: list[tuple[Any, dict[Any, Any]]] = claim[1]  # (next_id, messages, deleted)
        for msg_id, fields in claimed:
            self.stats.reclaimed += 1
            log.warning("reclaimed job %s from a stalled worker", _s(msg_id))
            await self._handle(msg_id, fields)
            handled += 1
        resp: Any = await self.q.redis.xreadgroup(
            self.group, self.consumer, {self.q.stream: ">"}, count=1, block=block_ms
        )
        for _, messages in resp or []:
            for msg_id, fields in messages:
                await self._handle(msg_id, fields)
                handled += 1
        return handled

    async def run_forever(
        self, stop: asyncio.Event, tick: Callable[[], Awaitable[None]] | None = None
    ) -> None:
        await self.setup()
        while not stop.is_set():
            if tick is not None:
                await tick()  # e.g. the daily scheduler
            await self.run_once()

    async def _handle(self, msg_id: bytes | str, fields: dict[Any, Any]) -> None:
        raw = fields.get(b"job") or fields.get("job")
        job: Payload = json.loads(_s(raw)) if raw else {"name": "?", "payload": {}, "attempt": 1}
        name, attempt = job.get("name", "?"), int(job.get("attempt", 1))
        handler = self.handlers.get(name)
        try:
            if handler is None:
                raise LookupError(f"no handler for job {name!r}")
            log.info("job %s attempt %d started", name, attempt)
            await handler(job.get("payload", {}), self.q)
            self.stats.done += 1
            log.info("job %s done", name)
        except Exception as e:
            retryable = handler is not None and not isinstance(e, PermanentError)
            if retryable and attempt < self.max_attempts:
                delay = self.base_delay_s * 2 ** (attempt - 1)
                log.warning(
                    "job %s failed (attempt %d), retry in %.0f s: %s", name, attempt, delay, e
                )
                await self.q._push(
                    {**job, "attempt": attempt + 1, "last_error": str(e)[:500]}, delay
                )
                self.stats.retried += 1
            else:
                log.error("job %s dead after %d attempt(s): %s", name, attempt, e)
                dead = {**job, "error": f"{type(e).__name__}: {e}"[:1000], "failed_at": time.time()}
                await self.q.redis.xadd(self.q.dead, {"job": json.dumps(dead)}, maxlen=1000)
                self.stats.dead += 1
        # Ack only here: the job succeeded, or was re-queued / dead-lettered. If the
        # worker is cancelled or crashes mid-job we never get here, the message stays
        # pending, and another worker reclaims it after the visibility timeout.
        await self.q.redis.xack(self.q.stream, self.group, msg_id)
