"""Hybrid retrieval against the test database, using fake (deterministic) models."""

from datetime import date

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import (
    EMBEDDING_DIM,
    Chunk,
    ContentStatus,
    Document,
    DocumentVersion,
)
from app.embeddings.base import HashEmbedder
from app.embeddings.pipeline import embed_pending_chunks
from app.retrieval.hybrid import HybridRetriever, SearchFilters
from app.retrieval.rerank import OverlapReranker

pytestmark = pytest.mark.integration

OLD_FEE = "5. The processing fee for a loan shall not exceed two per cent of the amount."
NEW_FEE = "5. The processing fee for a loan shall not exceed one per cent of the amount."
KYC = "9. Video based customer identification process V-CIP may be used for onboarding."


async def _seed(factory: async_sessionmaker[AsyncSession]) -> None:
    async with factory() as s, s.begin():
        fees = Document(
            regulator="RBI",
            doc_type="master_direction",
            source_key="t-fees",
            title="Fees Directions",
            ref_numbers=[],
            source_url="https://e/fees",
        )
        kyc = Document(
            regulator="RBI",
            doc_type="master_direction",
            source_key="t-kyc",
            title="KYC Directions",
            ref_numbers=[],
            source_url="https://e/kyc",
        )
        s.add_all([fees, kyc])
        await s.flush()

        def version(
            doc: Document, start: date, end: date | None, status: ContentStatus
        ) -> DocumentVersion:
            v = DocumentVersion(
                document_id=doc.id,
                valid_from=start,
                valid_to=end,
                title=doc.title,
                content_status=status,
                annex_links=[],
            )
            s.add(v)
            return v

        v_old = version(fees, date(2025, 1, 1), date(2026, 1, 1), ContentStatus.CAPTURED)
        v_new = version(fees, date(2026, 1, 1), None, ContentStatus.CAPTURED)
        v_kyc_gap = version(kyc, date(2024, 1, 1), date(2025, 6, 1), ContentStatus.UNAVAILABLE)
        v_kyc = version(kyc, date(2025, 6, 1), None, ContentStatus.CAPTURED)
        await s.flush()

        for v, txt, title in [
            (v_old, OLD_FEE, "Fees Directions"),
            (v_new, NEW_FEE, "Fees Directions"),
            (v_kyc, KYC, "KYC Directions"),
        ]:
            s.add(
                Chunk(
                    version_id=v.id,
                    ordinal=0,
                    kind="text",
                    heading_path=["Chapter I"],
                    para_refs=[txt.split(".")[0]],
                    context=f"{title} > Chapter I",
                    text=txt,
                    token_count=20,
                    amendments=[],
                    chunker_version="test",
                )
            )
        _ = v_kyc_gap


@pytest.fixture
async def retriever(
    committing_session_factory: async_sessionmaker[AsyncSession],
) -> HybridRetriever:
    session_factory = committing_session_factory
    await _seed(session_factory)
    embedder = HashEmbedder(dim=EMBEDDING_DIM)
    stats = await embed_pending_chunks(session_factory, embedder)
    assert stats.embedded == 3
    return HybridRetriever(session_factory, embedder, OverlapReranker())


async def test_finds_the_relevant_clause(retriever: HybridRetriever) -> None:
    hits, trace = await retriever.search("video customer identification", k=3)
    assert hits[0].document_title == "KYC Directions"
    assert hits[0].para_refs == ["9"]
    assert hits[0].vector_rank is not None and hits[0].keyword_rank is not None
    assert trace.vector_candidates >= 1 and trace.keyword_candidates >= 1
    assert {"embed_query", "vector_search", "total"} <= set(trace.timings_ms)


async def test_as_of_date_selects_the_version_in_force(retriever: HybridRetriever) -> None:
    today, _ = await retriever.search("loan processing fee", k=5)
    past, _ = await retriever.search(
        "loan processing fee", k=5, filters=SearchFilters(as_of=date(2025, 6, 30))
    )
    assert [h.text for h in today if "processing fee" in h.text] == [NEW_FEE]
    assert [h.text for h in past if "processing fee" in h.text] == [OLD_FEE]
    assert past[0].version_valid_to == date(2026, 1, 1)


async def test_versions_without_captured_text_are_never_returned(
    retriever: HybridRetriever,
) -> None:
    hits, _ = await retriever.search(
        "video customer identification", k=5, filters=SearchFilters(as_of=date(2024, 6, 1))
    )
    assert all(h.document_title != "KYC Directions" for h in hits)


async def test_regulator_filter(retriever: HybridRetriever) -> None:
    hits, _ = await retriever.search("processing fee", filters=SearchFilters(regulators=("SEBI",)))
    assert hits == []


async def test_embedding_is_idempotent(
    committing_session_factory: async_sessionmaker[AsyncSession], retriever: HybridRetriever
) -> None:
    again = await embed_pending_chunks(committing_session_factory, HashEmbedder(dim=EMBEDDING_DIM))
    assert again.embedded == 0
