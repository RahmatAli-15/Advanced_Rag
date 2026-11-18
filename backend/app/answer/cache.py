"""Exact-match answer cache in Redis.

Why only exact match: a "semantic" cache (similar question -> same answer) is risky in
compliance, where "Can NBFCs do video KYC?" and "Can banks do video KYC?" are close in
embedding space but have different answers. Exact match after light normalisation
(case, spacing, trailing "?") is safe and still catches repeated questions.

The key includes everything an answer depends on:
    normalised question · as-of date · prompt version · corpus fingerprint
The corpus fingerprint (counts and latest timestamps of versions, chunks and
embeddings) changes whenever a crawl, re-chunk or re-embed changes what retrieval can
see, so stale answers are never served and nobody has to remember to clear the cache.

Only settled outcomes are cached (answered, insufficient_sources, not_found,
no_coverage). Transient failures (llm_unavailable, ungrounded) are not.
If Redis is down the cache is skipped: it can make answers faster, never break them.
"""

import hashlib
import logging
import re
import time
import unicodedata
from collections.abc import Callable
from datetime import date

from pydantic import TypeAdapter
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.answer.service import AnswerResult
from app.db.models import Chunk, ChunkEmbedding, DocumentVersion

log = logging.getLogger(__name__)

CACHEABLE = frozenset({"answered", "insufficient_sources", "not_found", "no_coverage"})
_RESULT = TypeAdapter(AnswerResult)


def normalise_question(question: str) -> str:
    q = unicodedata.normalize("NFKC", question).casefold()
    q = re.sub(r"\s+", " ", q).strip()
    return q.rstrip(" ?.!।")


class CorpusFingerprint:
    """A short string that changes whenever the searchable corpus changes. Computed
    with three cheap aggregate queries, remembered for `ttl_s` seconds."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        ttl_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sf, self._ttl, self._clock = session_factory, ttl_s, clock
        self._value: str | None = None
        self._at = 0.0

    async def get(self) -> str:
        if self._value is not None and self._clock() - self._at < self._ttl:
            return self._value
        async with self._sf() as s:
            versions, recorded = (
                await s.execute(
                    select(func.count(DocumentVersion.id), func.max(DocumentVersion.recorded_at))
                )
            ).one()
            chunks, chunked = (
                await s.execute(select(func.count(Chunk.id), func.max(Chunk.created_at)))
            ).one()
            embeddings = await s.scalar(select(func.count()).select_from(ChunkEmbedding))
        raw = f"{versions}|{recorded}|{chunks}|{chunked}|{embeddings}"
        self._value = hashlib.sha256(raw.encode()).hexdigest()[:16]
        self._at = self._clock()
        return self._value


class AnswerCache:
    def __init__(
        self,
        redis: Redis,
        fingerprint: CorpusFingerprint,
        ttl_s: int = 24 * 3600,
        prefix: str = "answer:v1:",
    ) -> None:
        self._redis, self._fp = redis, fingerprint
        self._ttl, self._prefix = ttl_s, prefix
        self.hits = 0
        self.misses = 0

    async def key(self, question: str, as_of: date, prompt_version: str) -> str:
        corpus = await self._fp.get()
        raw = "\x1f".join([normalise_question(question), as_of.isoformat(), prompt_version, corpus])
        return self._prefix + hashlib.sha256(raw.encode()).hexdigest()

    async def get(self, key: str) -> AnswerResult | None:
        try:
            raw = await self._redis.get(key)
        except RedisError as e:
            log.warning("answer cache unavailable, skipping: %s", e)
            return None
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return _RESULT.validate_json(raw)

    async def set(self, key: str, result: AnswerResult) -> None:
        if result.status not in CACHEABLE:
            return
        try:
            await self._redis.set(key, _RESULT.dump_json(result), ex=self._ttl)
        except RedisError as e:
            log.warning("answer cache write failed: %s", e)
