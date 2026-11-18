"""POST /search — hybrid retrieval over regulations in force on a date."""

import uuid
from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.api.ratelimit import rate_limit
from app.retrieval.hybrid import HybridRetriever, SearchFilters

router = APIRouter(tags=["search"], dependencies=[Depends(rate_limit)])


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    k: int = Field(default=8, ge=1, le=20)
    as_of: date | None = Field(
        default=None, description="Search the rules in force on this date (default: today)"
    )
    regulators: list[str] = Field(default_factory=list)


class Hit(BaseModel):
    chunk_id: uuid.UUID
    document_title: str
    regulator: str
    source_url: str
    version_valid_from: date
    version_valid_to: date | None
    heading_path: list[str]
    para_refs: list[str]
    text: str
    amendments: list[dict[str, Any]]
    scores: dict[str, float | int | None]


class SearchResponse(BaseModel):
    as_of: date
    hits: list[Hit]
    trace: dict[str, Any]


@router.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, request: Request) -> SearchResponse:
    retriever: HybridRetriever = request.app.state.retriever
    hits, trace = await retriever.search(
        body.query,
        k=body.k,
        filters=SearchFilters(as_of=body.as_of, regulators=tuple(body.regulators)),
    )
    return SearchResponse(
        as_of=trace.as_of,
        hits=[
            Hit(
                chunk_id=h.chunk_id,
                document_title=h.document_title,
                regulator=h.regulator,
                source_url=h.source_url,
                version_valid_from=h.version_valid_from,
                version_valid_to=h.version_valid_to,
                heading_path=h.heading_path,
                para_refs=h.para_refs,
                text=h.text,
                amendments=h.amendments,
                scores={
                    "vector_rank": h.vector_rank,
                    "keyword_rank": h.keyword_rank,
                    "fused": h.fused_score,
                    "rerank": h.rerank_score,
                },
            )
            for h in hits
        ],
        trace={
            "timings_ms": trace.timings_ms,
            "vector_candidates": trace.vector_candidates,
            "keyword_candidates": trace.keyword_candidates,
            "fused_candidates": trace.fused_candidates,
            "reranked": trace.reranked,
        },
    )
