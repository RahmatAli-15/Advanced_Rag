"""Hybrid retrieval: vector + keyword search, fused with RRF, then reranked.

    query ─┬─▶ keyword search (Postgres full-text, OR of terms) ───────┐
           └─▶ embed (GPU, worker thread) ─▶ vector search (HNSW) ─────┴─▶ RRF ─▶ rerank ─▶ top k

Keyword search starts while the query is being embedded, so its latency is hidden.
Both searches see only chunks from document versions *in force on the as-of date*
(today by default). pgvector's iterative HNSW scan keeps returning neighbours until
enough rows pass that filter, instead of silently returning fewer results.
"""

import asyncio
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from sqlalchemy import ColumnElement, and_, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Chunk, ChunkEmbedding, ContentStatus, Document, DocumentVersion
from app.embeddings.base import Embedder
from app.retrieval.fusion import rrf_fuse
from app.retrieval.gate import ConcurrencyGate
from app.retrieval.rerank import Reranker

SearchMode = Literal["hybrid", "vector", "keyword"]


@dataclass(frozen=True)
class SearchFilters:
    as_of: date | None = None  # None = today
    regulators: tuple[str, ...] = ()
    document_ids: tuple[uuid.UUID, ...] = ()


@dataclass(frozen=True)
class SearchHit:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    regulator: str
    source_url: str
    version_valid_from: date
    version_valid_to: date | None
    heading_path: list[str]
    para_refs: list[str]
    text: str
    context: str
    amendments: list[dict[str, Any]]
    vector_rank: int | None
    keyword_rank: int | None
    fused_score: float
    rerank_score: float | None


@dataclass
class RetrievalTrace:
    """Per-stage numbers for debugging, evaluation and latency budgets."""

    as_of: date
    timings_ms: dict[str, float] = field(default_factory=dict)
    vector_candidates: int = 0
    keyword_candidates: int = 0
    fused_candidates: int = 0
    reranked: int = 0


def keyword_tsquery(query: str) -> str | None:
    """OR of the query's words. websearch_to_tsquery ANDs every word, so a natural
    question ("what are the KYC rules for NBFCs") would match almost nothing. With OR,
    ts_rank_cd still ranks chunks matching more (and closer) terms higher."""
    words = re.findall(r"[A-Za-z0-9]+", query)
    return " | ".join(dict.fromkeys(w.lower() for w in words)) or None


def _in_force(on: date, filters: SearchFilters) -> ColumnElement[bool]:
    conds: list[ColumnElement[bool]] = [
        DocumentVersion.content_status == ContentStatus.CAPTURED,
        DocumentVersion.valid_from <= on,
        or_(DocumentVersion.valid_to.is_(None), DocumentVersion.valid_to > on),
    ]
    if filters.regulators:
        conds.append(Document.regulator.in_(filters.regulators))
    if filters.document_ids:
        conds.append(Document.id.in_(filters.document_ids))
    return and_(*conds)


class HybridRetriever:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedder: Embedder,
        reranker: Reranker | None = None,
        *,
        candidates: int = 40,
        rerank_top: int = 30,
        ef_search: int = 100,
        keyword_weight: float = 0.3,  # measured: eval/reports, ADR 0006
        gate: ConcurrencyGate | None = None,  # load shedding for the API (ADR 0013)
    ) -> None:
        self._gate = gate
        self._sf = session_factory
        self._embedder = embedder
        self._reranker = reranker
        self._candidates = candidates
        self._rerank_top = rerank_top
        self._ef_search = ef_search
        self._keyword_weight = keyword_weight

    async def _keyword(self, query: str, on: date, filters: SearchFilters) -> list[uuid.UUID]:
        tsq_text = keyword_tsquery(query)
        if tsq_text is None:
            return []
        tsq = func.to_tsquery("english", tsq_text)
        stmt = (
            select(Chunk.id)
            .join(DocumentVersion, DocumentVersion.id == Chunk.version_id)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(_in_force(on, filters), Chunk.search_tsv.op("@@")(tsq))
            .order_by(func.ts_rank_cd(Chunk.search_tsv, tsq).desc())
            .limit(self._candidates)
        )
        async with self._sf() as s:
            return list((await s.scalars(stmt)).all())

    async def _vector(self, qvec: list[float], on: date, filters: SearchFilters) -> list[uuid.UUID]:
        stmt = (
            select(Chunk.id)
            .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
            .join(DocumentVersion, DocumentVersion.id == Chunk.version_id)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(ChunkEmbedding.model == self._embedder.name, _in_force(on, filters))
            .order_by(ChunkEmbedding.embedding.cosine_distance(qvec))
            .limit(self._candidates)
        )
        async with self._sf() as s, s.begin():
            # SET LOCAL lasts only for this transaction.
            await s.execute(text(f"SET LOCAL hnsw.ef_search = {int(self._ef_search)}"))
            await s.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            return list((await s.scalars(stmt)).all())

    async def _load(self, ids: list[uuid.UUID]) -> dict[uuid.UUID, Any]:
        stmt = (
            select(Chunk, DocumentVersion, Document)
            .join(DocumentVersion, DocumentVersion.id == Chunk.version_id)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(Chunk.id.in_(ids))
        )
        async with self._sf() as s:
            return {row[0].id: row for row in (await s.execute(stmt)).all()}

    async def search(
        self,
        query: str,
        k: int = 8,
        filters: SearchFilters | None = None,
        mode: SearchMode = "hybrid",
        keyword_weight: float | None = None,
    ) -> tuple[list[SearchHit], RetrievalTrace]:
        """`mode` exists for evaluation: it isolates each retriever's contribution.
        With a gate, raises Overloaded when the models are saturated."""
        if self._gate is None:
            return await self._search(query, k, filters, mode, keyword_weight)
        async with self._gate.slot():
            return await self._search(query, k, filters, mode, keyword_weight)

    async def _search(
        self,
        query: str,
        k: int,
        filters: SearchFilters | None,
        mode: SearchMode,
        keyword_weight: float | None,
    ) -> tuple[list[SearchHit], RetrievalTrace]:
        filters = filters or SearchFilters()
        on = filters.as_of or date.today()
        trace = RetrievalTrace(as_of=on)
        t0 = time.perf_counter()

        def mark(stage: str, since: float) -> float:
            now = time.perf_counter()
            trace.timings_ms[stage] = round((now - since) * 1000, 1)
            return now

        keyword_task = (
            asyncio.create_task(self._keyword(query, on, filters)) if mode != "vector" else None
        )
        vector_ids: list[uuid.UUID] = []
        if mode != "keyword":
            t = time.perf_counter()
            qvec = await asyncio.to_thread(self._embedder.embed_query, query)
            t = mark("embed_query", t)
            vector_ids = await self._vector(qvec, on, filters)
            mark("vector_search", t)
        keyword_ids = await keyword_task if keyword_task else []
        mark("retrieve", t0)
        trace.vector_candidates, trace.keyword_candidates = len(vector_ids), len(keyword_ids)

        kw = self._keyword_weight if keyword_weight is None else keyword_weight
        fused = rrf_fuse([vector_ids, keyword_ids], weights=[1.0, kw])
        trace.fused_candidates = len(fused)
        vrank = {cid: i for i, cid in enumerate(vector_ids, start=1)}
        krank = {cid: i for i, cid in enumerate(keyword_ids, start=1)}

        shortlist = fused[: self._rerank_top if self._reranker else k]
        t = time.perf_counter()
        rows = await self._load([cid for cid, _ in shortlist])
        t = mark("load", t)

        rerank_scores: dict[uuid.UUID, float] = {}
        if self._reranker and shortlist:
            passages = [f"{rows[cid][0].context}\n{rows[cid][0].text}" for cid, _ in shortlist]
            scores = await asyncio.to_thread(self._reranker.score, query, passages)
            rerank_scores = {cid: s for (cid, _), s in zip(shortlist, scores, strict=True)}
            shortlist = sorted(shortlist, key=lambda x: rerank_scores[x[0]], reverse=True)
            trace.reranked = len(passages)
            mark("rerank", t)

        hits = []
        for cid, fused_score in shortlist[:k]:
            chunk, version, doc = rows[cid]
            hits.append(
                SearchHit(
                    chunk_id=cid,
                    document_id=doc.id,
                    document_title=doc.title,
                    regulator=doc.regulator,
                    source_url=doc.source_url,
                    version_valid_from=version.valid_from,
                    version_valid_to=version.valid_to,
                    heading_path=chunk.heading_path,
                    para_refs=chunk.para_refs,
                    text=chunk.text,
                    context=chunk.context,
                    amendments=chunk.amendments,
                    vector_rank=vrank.get(cid),
                    keyword_rank=krank.get(cid),
                    fused_score=round(fused_score, 5),
                    rerank_score=rerank_scores.get(cid),
                )
            )
        trace.timings_ms["total"] = round((time.perf_counter() - t0) * 1000, 1)
        return hits, trace
