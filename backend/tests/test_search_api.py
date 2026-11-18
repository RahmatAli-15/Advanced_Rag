"""API contract for POST /search, with the retriever stubbed out."""

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import create_app
from app.retrieval.hybrid import RetrievalTrace, SearchFilters, SearchHit


class StubRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, SearchFilters]] = []

    async def search(
        self, query: str, k: int = 8, filters: SearchFilters | None = None
    ) -> tuple[list[SearchHit], RetrievalTrace]:
        f = filters or SearchFilters()
        self.calls.append((query, k, f))
        hit = SearchHit(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            document_title="KYC Directions",
            regulator="RBI",
            source_url="https://e",
            version_valid_from=date(2025, 6, 1),
            version_valid_to=None,
            heading_path=["Chapter I"],
            para_refs=["9"],
            text="9. V-CIP",
            context="KYC",
            amendments=[],
            vector_rank=1,
            keyword_rank=2,
            fused_score=0.03,
            rerank_score=4.2,
        )
        return [hit], RetrievalTrace(as_of=f.as_of or date(2026, 11, 18), timings_ms={"total": 5})


@asynccontextmanager
async def _noop(_: FastAPI) -> AsyncIterator[None]:
    yield


def _client() -> tuple[TestClient, StubRetriever]:
    app = create_app()
    app.router.lifespan_context = _noop
    stub = StubRetriever()
    app.state.retriever = stub
    return TestClient(app), stub


def test_search_returns_hits_with_citation_fields() -> None:
    client, stub = _client()
    r = client.post("/search", json={"query": "video KYC", "k": 3, "as_of": "2026-03-01"})
    assert r.status_code == 200
    body: dict[str, Any] = r.json()
    assert body["as_of"] == "2026-03-01"
    hit = body["hits"][0]
    assert hit["para_refs"] == ["9"] and hit["document_title"] == "KYC Directions"
    assert hit["scores"] == {"vector_rank": 1, "keyword_rank": 2, "fused": 0.03, "rerank": 4.2}
    assert stub.calls[0][1] == 3 and stub.calls[0][2].as_of == date(2026, 3, 1)


def test_search_validates_input() -> None:
    client, _ = _client()
    assert client.post("/search", json={"query": "x"}).status_code == 422
    assert client.post("/search", json={"query": "valid", "k": 100}).status_code == 422
