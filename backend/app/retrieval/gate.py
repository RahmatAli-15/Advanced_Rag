"""Load shedding for the GPU models.

The load test showed one GPU serves ~1.3-1.5 searches per second whatever the
concurrency: more parallel requests only make a longer queue (p95 0.9 s with 1 user,
8.8 s with 8).

So: at most `max_concurrent` searches use the models at once, at most `max_waiting`
wait for a turn, and anyone beyond that is refused at once (HTTP 503 + Retry-After)
instead of joining a queue they would time out in. Fast failure lets a client retry
or back off; a 9-second wait helps nobody.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.core import metrics


class Overloaded(Exception):
    # 1 s: about one search's service time. Longer leaves the GPU idle between retries.
    def __init__(self, retry_after_s: int = 1) -> None:
        super().__init__("too many requests in progress; retry shortly")
        self.retry_after_s = retry_after_s


class ConcurrencyGate:
    def __init__(self, max_concurrent: int, max_waiting: int) -> None:
        self.max_concurrent, self.max_waiting = max_concurrent, max_waiting
        self._sem = asyncio.Semaphore(max_concurrent)
        self._running = 0
        self._waiting = 0

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        if self._sem.locked() and self._waiting >= self.max_waiting:
            metrics.SHED.inc()
            raise Overloaded()
        self._waiting += 1
        metrics.GATE.labels("waiting").set(self._waiting)
        try:
            await self._sem.acquire()
        finally:
            self._waiting -= 1
            metrics.GATE.labels("waiting").set(self._waiting)
        self._running += 1
        metrics.GATE.labels("running").set(self._running)
        try:
            yield
        finally:
            self._running -= 1
            metrics.GATE.labels("running").set(self._running)
            self._sem.release()
