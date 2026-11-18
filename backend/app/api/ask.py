"""POST /ask — a cited answer from the regulations in force on a date."""

import uuid
from datetime import date
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.answer.service import AnswerService, Status

router = APIRouter(tags=["ask"])


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    as_of: date | None = Field(default=None, description="Rules in force on this date")


class CitationOut(BaseModel):
    id: str
    document_title: str
    para_refs: list[str]
    source_url: str
    version_valid_from: date
    amendments: list[dict[str, Any]]
    quote: str


class AskResponse(BaseModel):
    status: Status
    answer: str
    citations: list[CitationOut]
    as_of: date
    model: str | None
    timings_ms: dict[str, float]
    log_id: uuid.UUID | None


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    service: AnswerService | None = getattr(request.app.state, "answer_service", None)
    if service is None:
        raise HTTPException(503, "answering is not configured (GROQ_API_KEY missing)")
    r = await service.ask(body.question, body.as_of)
    return AskResponse(
        status=r.status,
        answer=r.answer,
        citations=[
            CitationOut(
                id=c.id,
                document_title=c.document_title,
                para_refs=c.para_refs,
                source_url=c.source_url,
                version_valid_from=c.version_valid_from,
                amendments=c.amendments,
                quote=c.quote,
            )
            for c in r.citations
        ],
        as_of=r.as_of,
        model=r.model,
        timings_ms=r.timings_ms,
        log_id=r.log_id,
    )
