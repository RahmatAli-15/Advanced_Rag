"""Daily token budget for the primary (most expensive) model.

When today's tokens on the primary model pass the limit, the gateway starts at the next,
cheaper model instead (graceful degradation: answers keep coming, at lower cost). The
counter lives in Redis so the API and the worker share one budget, and resets at UTC
midnight, when the provider's daily quotas reset too.

The budget is a cost control, not a correctness feature: if Redis is down it is ignored
(never blocks an answer).
"""

import logging
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

from app.core import metrics

log = logging.getLogger(__name__)


def _today_utc() -> date:
    return datetime.now(UTC).date()


class DailyTokenBudget:
    def __init__(
        self,
        redis: Any,
        limit: int,
        *,
        prefix: str = "llm:budget:",
        today: Callable[[], date] = _today_utc,
    ) -> None:
        self.redis, self.limit, self.prefix, self._today = redis, limit, prefix, today
        metrics.LLM_BUDGET_LIMIT.set(limit)

    def _key(self) -> str:
        return f"{self.prefix}{self._today().isoformat()}"

    async def used(self) -> int:
        return int(await self.redis.get(self._key()) or 0)

    async def exhausted(self) -> bool:
        if self.limit <= 0:
            return False
        try:
            used = await self.used()
        except Exception as e:
            log.warning("token budget unavailable, ignoring it: %s", e)
            return False
        metrics.LLM_BUDGET_USED.set(used)
        return used >= self.limit

    async def add(self, tokens: int) -> None:
        if tokens <= 0:
            return
        try:
            key = self._key()
            used = await self.redis.incrby(key, tokens)
            await self.redis.expire(key, 2 * 24 * 3600)  # keep yesterday for debugging
        except Exception as e:
            log.warning("could not record %d tokens in the budget: %s", tokens, e)
            return
        metrics.LLM_BUDGET_USED.set(used)
