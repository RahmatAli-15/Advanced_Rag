"""Chunking pipeline against the test database: raw HTML in, searchable chunks out."""

from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.chunking.chunker import CHUNKER_VERSION
from app.chunking.pipeline import chunk_versions
from app.db.models import Chunk
from app.ingestion import repository as repo
from app.ingestion.raw_store import RawStore

pytestmark = pytest.mark.integration
FIXTURE = Path(__file__).parent / "fixtures" / "rbi_detail.html"


async def _seed(factory: async_sessionmaker[AsyncSession], store: RawStore) -> None:
    blob = store.put(FIXTURE.read_bytes(), ".html")
    async with factory() as s, s.begin():
        doc_id, _ = await repo.upsert_document(
            s,
            repo.DocumentData(
                "RBI",
                "master_direction",
                "chunk-test",
                "CRR and SLR Directions",
                None,
                [],
                None,
                "https://example.org",
            ),
        )
        await repo.record_captured_version(
            s,
            doc_id,
            repo.CapturedVersion(
                date(2026, 10, 9),
                "CRR and SLR Directions",
                blob.sha256,
                blob.relative_path,
                "fp",
                7,
                [],
            ),
        )


async def test_chunks_are_written_searchable_and_idempotent(
    session_factory: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    store = RawStore(tmp_path)
    await _seed(session_factory, store)

    first = await chunk_versions(session_factory, store)
    again = await chunk_versions(session_factory, store)
    rebuilt = await chunk_versions(session_factory, store, rebuild=True)

    assert first.versions == 1 and first.chunks > 0
    assert (again.versions, again.chunks) == (0, 0)
    assert rebuilt.chunks == first.chunks  # rebuild replaces, never duplicates

    async with session_factory() as s:
        n = await s.scalar(select(func.count()).select_from(Chunk))
        assert n == first.chunks
        hit = await s.scalar(
            select(Chunk).where(
                Chunk.search_tsv.op("@@")(text("websearch_to_tsquery('english', 'treasury bills')"))
            )
        )
        assert hit is not None and hit.chunker_version == CHUNKER_VERSION
        assert hit.context.startswith("CRR and SLR Directions")

        amended = await s.scalar(select(Chunk).where(func.jsonb_array_length(Chunk.amendments) > 0))
        assert amended is not None
        assert {a["id"] for a in amended.amendments} >= {"F1"}
