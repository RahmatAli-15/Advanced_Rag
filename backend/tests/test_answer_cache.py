"""Answer cache: key normalisation, hits and audit, what is (not) cached, invalidation,
and that a broken cache never breaks answering."""

import json
from datetime import date

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.answer.cache import AnswerCache, CorpusFingerprint, normalise_question
from app.answer.service import AnswerService, FinalEvent
from app.db.models import AnswerLog, ContentStatus, Document, DocumentVersion
from app.llm.gateway import LLMGateway, ModelRoute
from tests.test_answer import HITS, FakeLLM, StubRetriever
from tests.test_streaming import StreamingProvider


class FakeRedis:
    def __init__(self, broken: bool = False) -> None:
        self.data: dict[str, bytes] = {}
        self.broken = broken

    async def get(self, key: str) -> bytes | None:
        if self.broken:
            raise RedisConnectionError("down")
        return self.data.get(key)

    async def set(self, key: str, value: bytes, ex: int | None = None) -> None:
        if self.broken:
            raise RedisConnectionError("down")
        self.data[key] = value


class FixedFingerprint:
    def __init__(self, value: str = "corpus-1") -> None:
        self.value = value

    async def get(self) -> str:
        return self.value


def _cache(redis: FakeRedis | None = None, fp: FixedFingerprint | None = None) -> AnswerCache:
    return AnswerCache(redis or FakeRedis(), fp or FixedFingerprint())  # type: ignore[arg-type]


def _reply(answerable: bool, answer: str) -> str:
    return json.dumps({"answerable": answerable, "answer": answer})


def _service(
    factory: async_sessionmaker[AsyncSession], llm: FakeLLM, cache: AnswerCache
) -> AnswerService:
    gateway = LLMGateway([ModelRoute(llm, "m1", 6000)], max_retries=0)
    return AnswerService(StubRetriever(HITS), gateway, factory, cache=cache)  # type: ignore[arg-type]


def test_question_normalisation_is_conservative() -> None:
    assert normalise_question("  How often  must KYC be UPDATED? ") == normalise_question(
        "how often must kyc be updated"
    )
    assert normalise_question("क्या KYC ज़रूरी है?") == normalise_question("क्या kyc ज़रूरी है")
    # different words are different questions: no "semantic" matching
    assert normalise_question("Can NBFCs do video KYC?") != normalise_question(
        "Can banks do video KYC?"
    )


async def test_key_changes_with_date_prompt_and_corpus() -> None:
    fp = FixedFingerprint()
    cache = _cache(fp=fp)
    base = await cache.key("q?", date(2026, 1, 1), "answer-v2")
    assert base == await cache.key("Q", date(2026, 1, 1), "answer-v2")
    assert base != await cache.key("q", date(2025, 1, 1), "answer-v2")
    assert base != await cache.key("q", date(2026, 1, 1), "answer-v3")
    fp.value = "corpus-2"  # a crawl or re-embed happened
    assert base != await cache.key("q", date(2026, 1, 1), "answer-v2")


@pytest.mark.integration
async def test_repeat_question_is_served_from_cache_and_still_audited(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    llm = FakeLLM([_reply(True, "Every two years [S1].")])
    service = _service(session_factory, llm, _cache())

    first = await service.ask("KYC update for high risk?")
    second = await service.ask("kyc update for HIGH risk")

    assert len(llm.requests) == 1  # the LLM was called once
    assert not first.cached and second.cached
    assert second.answer == first.answer and second.citations == first.citations
    assert second.usage == {} and second.log_id != first.log_id
    async with session_factory() as s:
        row = await s.scalar(select(AnswerLog).where(AnswerLog.id == second.log_id))
    assert row is not None and row.cached


@pytest.mark.integration
async def test_transient_failures_are_not_cached(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    llm = FakeLLM([_reply(True, "No citation here."), _reply(True, "Two years [S1].")])
    service = _service(session_factory, llm, _cache())
    assert (await service.ask("q?")).status == "ungrounded"
    assert (await service.ask("q?")).status == "answered"  # asked the model again
    assert len(llm.requests) == 2


@pytest.mark.integration
async def test_broken_cache_never_breaks_answering(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    llm = FakeLLM([_reply(True, "Two years [S1].")])
    service = _service(session_factory, llm, _cache(FakeRedis(broken=True)))
    r = await service.ask("q?")
    assert r.status == "answered" and not r.cached


@pytest.mark.integration
async def test_stream_cache_hit_goes_straight_to_the_final_event(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    p = StreamingProvider({"m1": [["Every two years [S1]."]]})
    gateway = LLMGateway([ModelRoute(p, "m1", 6000)], max_retries=0)
    service = AnswerService(StubRetriever(HITS), gateway, session_factory, cache=_cache())  # type: ignore[arg-type]

    first = [e async for e in service.ask_stream("KYC?")]
    second = [e async for e in service.ask_stream("KYC?")]

    assert len(first) > 1 and p.calls == ["m1"]
    assert len(second) == 1 and isinstance(second[0], FinalEvent)
    assert second[0].result.cached and second[0].result.status == "answered"


@pytest.mark.integration
async def test_corpus_fingerprint_changes_when_a_version_is_added(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    fp = CorpusFingerprint(session_factory, ttl_s=0)
    before = await fp.get()
    async with session_factory() as s, s.begin():
        doc = Document(
            regulator="RBI",
            doc_type="master_direction",
            source_key="cache-test",
            title="T",
            source_url="https://e",
        )
        s.add(doc)
        await s.flush()
        s.add(
            DocumentVersion(
                document_id=doc.id,
                valid_from=date(2026, 1, 1),
                title="T",
                content_status=ContentStatus.UNAVAILABLE,
            )
        )
    assert await fp.get() != before
