"""Streaming: gateway retry-before-first-token policy, Groq SSE parsing, and the
streamed answer pipeline (events, NOT_IN_SOURCES marker, interruption, HTTP framing)."""

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.answer.service import (
    STREAM_PROMPT_VERSION,
    AnswerEvent,
    AnswerResult,
    AnswerService,
    DeltaEvent,
    FinalEvent,
    SourcesEvent,
    StageEvent,
)
from app.db.models import AnswerLog
from app.llm.gateway import LLMGateway, ModelRoute
from app.llm.groq import GroqProvider
from app.llm.types import (
    AllModelsFailed,
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
from app.main import create_app
from tests.test_answer import HITS, StubRetriever

REQ = LLMRequest([Message("user", "hi")])

# A script item is either an error raised before any text, or a list of deltas; a
# delta that is an LLMError is raised mid-stream (after earlier text went out).
Script = LLMError | list[str | LLMError]


class StreamingProvider:
    name = "fake"

    def __init__(self, script: dict[str, list[Script]]) -> None:
        self.script = script
        self.calls: list[str] = []
        self.requests: list[LLMRequest] = []
        self.closed = 0

    async def complete(self, request: LLMRequest, model: str) -> LLMResponse:
        raise NotImplementedError

    async def stream(self, request: LLMRequest, model: str) -> AsyncGenerator[StreamEvent]:
        self.calls.append(model)
        self.requests.append(request)
        outcome = self.script[model].pop(0)
        try:
            if isinstance(outcome, LLMError):
                raise outcome
            for piece in outcome:
                if isinstance(piece, LLMError):
                    raise piece
                yield StreamEvent(text=piece)
            yield StreamEvent(done=True, model=model, usage=Usage(50, 7))
        finally:
            self.closed += 1


async def _no_sleep(_: float) -> None:
    return None


def _gateway(p: StreamingProvider, models: list[str], **kw: Any) -> LLMGateway:
    return LLMGateway(
        [ModelRoute(p, m, requests_per_minute=6000) for m in models],
        sleep=_no_sleep,
        **kw,
    )


async def _collect(gen: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [e async for e in gen]


# --- gateway -------------------------------------------------------------------------


async def test_stream_yields_deltas_then_done_with_usage() -> None:
    p = StreamingProvider({"a": [["Hel", "lo"]]})
    gw = _gateway(p, ["a"])
    events = await _collect(gw.stream(REQ))
    assert [e.text for e in events if not e.done] == ["Hel", "lo"]
    assert events[-1].done and events[-1].model == "a"
    assert gw.stats.usage_by_model["a"] == Usage(50, 7)


async def test_stream_retries_and_falls_back_before_the_first_token() -> None:
    p = StreamingProvider(
        {"a": [RateLimited("429", retry_after_s=1), TransientError("500")], "b": [["from b"]]}
    )
    gw = _gateway(p, ["a", "b"], max_retries=1, breaker_threshold=10)
    events = await _collect(gw.stream(REQ))
    assert p.calls == ["a", "a", "b"]
    assert events[0].text == "from b"
    assert gw.stats.retries == 1 and gw.stats.fallbacks == 1


async def test_stream_failure_after_first_token_is_not_retried() -> None:
    p = StreamingProvider({"a": [["partial", TransientError("reset")]], "b": [["never"]]})
    gw = _gateway(p, ["a", "b"])
    seen: list[str] = []
    with pytest.raises(StreamInterrupted):
        async for e in gw.stream(REQ):
            seen.append(e.text)
    assert seen == ["partial"] and p.calls == ["a"]  # no splice of a second model


async def test_rate_limits_do_not_open_the_circuit_breaker() -> None:
    # Regression: three 429s in a row used to open the breaker, so a healthy but
    # busy judge model was skipped for every later request.
    p = StreamingProvider({"a": [RateLimited("429", retry_after_s=1)] * 5 + [["ok"]]})
    gw = _gateway(p, ["a"], max_retries=5, breaker_threshold=3)
    events = await _collect(gw.stream(REQ))
    assert events[0].text == "ok" and gw.breaker_state("a") == "closed"
    assert p.calls == ["a"] * 6


async def test_stream_raises_all_models_failed_when_none_start() -> None:
    p = StreamingProvider({"a": [TransientError("x")] * 2})
    with pytest.raises(AllModelsFailed):
        await _collect(_gateway(p, ["a"], max_retries=1).stream(REQ))


async def test_closing_the_consumer_closes_the_provider_stream() -> None:
    p = StreamingProvider({"a": [["one", "two", "three"]]})
    gen = _gateway(p, ["a"]).stream(REQ)
    assert (await anext(gen)).text == "one"
    await gen.aclose()  # e.g. the browser disconnected
    assert p.closed == 1


# --- Groq SSE parsing ----------------------------------------------------------------


def _sse_body(*chunks: dict[str, Any]) -> bytes:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


async def test_groq_stream_parses_deltas_and_usage() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        body = _sse_body(
            {"model": "m", "choices": [{"delta": {"role": "assistant", "content": ""}}]},
            {"model": "m", "choices": [{"delta": {"reasoning": "thinking..."}}]},
            {"model": "m", "choices": [{"delta": {"content": "Two "}}]},
            {"model": "m", "choices": [{"delta": {"content": "years [S1]"}}]},
            {"model": "m", "choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4}},
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    provider = GroqProvider("k", transport=httpx.MockTransport(handler))
    events = await _collect(provider.stream(REQ, "m"))
    assert [e.text for e in events if not e.done] == ["Two ", "years [S1]"]  # no reasoning
    assert events[-1] == StreamEvent(done=True, model="m", usage=Usage(9, 4))
    assert sent[0]["stream"] is True


async def test_groq_stream_maps_http_errors_before_any_text() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "3"}, text="slow down")

    provider = GroqProvider("k", transport=httpx.MockTransport(handler))
    with pytest.raises(RateLimited) as exc:
        await _collect(provider.stream(REQ, "m"))
    assert exc.value.retry_after_s == 3


async def test_groq_stream_enforces_a_total_deadline_on_a_trickling_stream() -> None:
    async def trickle() -> AsyncIterator[bytes]:
        for word in ["slow", "er", "and", "slower"]:
            chunk = {"model": "m", "choices": [{"delta": {"content": word}}]}
            yield f"data: {json.dumps(chunk)}\n\n".encode()
            await asyncio.sleep(0.05)  # each read is quick, so httpx never times out

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=trickle())

    provider = GroqProvider("k", transport=httpx.MockTransport(handler), max_stream_s=0.08)
    seen: list[str] = []
    with pytest.raises(TransientError, match="deadline"):
        async for e in provider.stream(REQ, "m"):
            seen.append(e.text)
    assert 0 < len(seen) < 4


# --- streamed answers ----------------------------------------------------------------


def _service(
    factory: async_sessionmaker[AsyncSession], script: list[Script], **kw: Any
) -> tuple[AnswerService, StreamingProvider]:
    p = StreamingProvider({"m1": script})
    gw = _gateway(p, ["m1"], max_retries=0)
    service = AnswerService(StubRetriever(kw.pop("hits", HITS), **kw), gw, factory)  # type: ignore[arg-type]
    return service, p


async def _events(service: AnswerService, q: str = "q?") -> list[AnswerEvent]:
    return [e async for e in service.ask_stream(q)]


def _final(events: list[AnswerEvent]) -> AnswerResult:
    last = events[-1]
    assert isinstance(last, FinalEvent)
    return last.result


@pytest.mark.integration
async def test_stream_emits_sources_then_deltas_then_verified_final(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    service, p = _service(session_factory, [["Every two", " years【S1】", " [S9]."]])
    events = await _events(service)

    kinds = [type(e).__name__ for e in events]
    assert kinds == [
        "StageEvent",
        "SourcesEvent",
        "StageEvent",
        "DeltaEvent",
        "DeltaEvent",
        "DeltaEvent",
        "FinalEvent",
    ]
    sources = events[1]
    assert isinstance(sources, SourcesEvent) and len(sources.citations) == 2
    assert sources.citations[0].text.startswith("(1) Periodic updation")
    result = _final(events)
    assert result.status == "answered"
    assert result.answer == "Every two years[S1] ."  # normalised, invented [S9] removed
    assert result.invalid_citations == ["S9"]
    assert "NOT_IN_SOURCES" in p.requests[0].messages[0].content
    async with session_factory() as s:
        row = await s.scalar(select(AnswerLog).where(AnswerLog.id == result.log_id))
    assert row is not None and row.prompt_version == STREAM_PROMPT_VERSION
    assert row.usage == {"prompt_tokens": 50, "completion_tokens": 7}


@pytest.mark.integration
async def test_marker_split_across_deltas_is_hidden_and_sets_insufficient(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    service, _ = _service(session_factory, [["NOT_IN", "_SOURCES\n", "No SEBI rules here."]])
    events = await _events(service)
    deltas = "".join(e.text for e in events if isinstance(e, DeltaEvent))
    assert "NOT_IN" not in deltas and deltas == "No SEBI rules here."
    assert _final(events).status == "insufficient_sources"


@pytest.mark.integration
async def test_short_answer_that_looks_like_marker_prefix_is_kept(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    service, _ = _service(session_factory, [["NO", " [S1]"]])
    result = _final(await _events(service))
    assert result.status == "answered" and result.answer == "NO [S1]"


@pytest.mark.integration
async def test_interrupted_stream_degrades_to_sources(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    service, _ = _service(session_factory, [["Every two", TransientError("reset")]])
    events = await _events(service)
    result = _final(events)
    assert result.status == "llm_unavailable" and len(result.citations) == 2


@pytest.mark.integration
async def test_low_score_stream_abstains_without_llm(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    from tests.test_answer import _hit

    service, p = _service(session_factory, [], hits=[_hit("unrelated", 0.01, ["1"])])
    events = await _events(service)
    assert [type(e) for e in events] == [StageEvent, FinalEvent]
    assert _final(events).status == "not_found" and p.calls == []


# --- HTTP framing --------------------------------------------------------------------


class StubAnswerService:
    async def ask_stream(self, question: str, as_of: date | None = None) -> AsyncIterator[Any]:
        yield StageEvent("retrieving")
        yield DeltaEvent("line one\nline two")
        yield FinalEvent(AnswerResult("answered", "line one\nline two [S1]", [], date(2026, 1, 1)))


def _parse_sse(raw: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for block in raw.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((fields["event"], json.loads(fields["data"])))
    return out


def test_ask_stream_endpoint_frames_events_as_sse() -> None:
    @asynccontextmanager
    async def noop(_: FastAPI) -> AsyncIterator[None]:
        yield

    app = create_app()
    app.router.lifespan_context = noop
    app.state.answer_service = StubAnswerService()
    with TestClient(app) as client:
        r = client.post("/ask/stream", json={"question": "KYC?"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(r.text)
    assert [e for e, _ in events] == ["stage", "delta", "final"]
    assert events[1][1]["text"] == "line one\nline two"  # newline survived framing
    assert events[2][1]["status"] == "answered"
