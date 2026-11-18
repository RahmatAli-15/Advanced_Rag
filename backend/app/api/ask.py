"""POST /ask — a cited answer from the regulations in force on a date.
POST /ask/stream — the same answer as Server-Sent Events (stage, sources, delta, final)."""

import json
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import date
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.answer.service import (
    AnswerResult,
    AnswerService,
    Citation,
    DeltaEvent,
    FinalEvent,
    SourcesEvent,
    StageEvent,
    Status,
)

log = logging.getLogger(__name__)
router = APIRouter(tags=["ask"])


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    as_of: date | None = Field(default=None, description="Rules in force on this date")


class CitationOut(BaseModel):
    id: str
    document_title: str
    para_refs: list[str]
    heading_path: list[str]
    source_url: str
    version_valid_from: date
    amendments: list[dict[str, Any]]
    quote: str
    text: str


class AskResponse(BaseModel):
    status: Status
    answer: str
    citations: list[CitationOut]
    as_of: date
    model: str | None
    timings_ms: dict[str, float]
    log_id: uuid.UUID | None


def _citation_out(c: Citation) -> CitationOut:
    return CitationOut(
        id=c.id,
        document_title=c.document_title,
        para_refs=c.para_refs,
        heading_path=c.heading_path,
        source_url=c.source_url,
        version_valid_from=c.version_valid_from,
        amendments=c.amendments,
        quote=c.quote,
        text=c.text,
    )


def _response(r: AnswerResult) -> AskResponse:
    return AskResponse(
        status=r.status,
        answer=r.answer,
        citations=[_citation_out(c) for c in r.citations],
        as_of=r.as_of,
        model=r.model,
        timings_ms=r.timings_ms,
        log_id=r.log_id,
    )


def _service(request: Request) -> AnswerService:
    service: AnswerService | None = getattr(request.app.state, "answer_service", None)
    if service is None:
        raise HTTPException(503, "answering is not configured (GROQ_API_KEY missing)")
    return service


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    r = await _service(request).ask(body.question, body.as_of)
    return _response(r)


def _sse(event: str, data: str) -> str:
    # data is JSON, so newlines in the answer are escaped and cannot break framing.
    return f"event: {event}\ndata: {data}\n\n"


@router.post(
    "/ask/stream",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def ask_stream(body: AskRequest, request: Request) -> StreamingResponse:
    service = _service(request)

    async def events() -> AsyncIterator[str]:
        # If the client disconnects, Starlette cancels this generator; the LLM stream
        # is closed with it, so we stop paying for tokens nobody will read.
        try:
            async for ev in service.ask_stream(body.question, body.as_of):
                match ev:
                    case StageEvent(stage=stage):
                        yield _sse("stage", json.dumps({"stage": stage}))
                    case SourcesEvent():
                        payload = {
                            "as_of": ev.as_of.isoformat(),
                            "retrieval_ms": ev.retrieval_ms,
                            "citations": [
                                _citation_out(c).model_dump(mode="json") for c in ev.citations
                            ],
                        }
                        yield _sse("sources", json.dumps(payload))
                    case DeltaEvent(text=text):
                        yield _sse("delta", json.dumps({"text": text}))
                    case FinalEvent(result=result):
                        yield _sse("final", _response(result).model_dump_json())
        except Exception:
            log.exception("ask stream failed")
            yield _sse("error", json.dumps({"message": "Something went wrong. Try again."}))

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
