"""Integration tests against a real Postgres (skipped when none is reachable).

Uses the dedicated test database from conftest; every test is rolled back.
"""

from datetime import date

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ContentStatus
from app.ingestion import repository as repo

pytestmark = pytest.mark.integration


DOC = repo.DocumentData(
    regulator="RBI",
    doc_type="master_direction",
    source_key="test-1",
    title="Test Directions, 2025",
    category="Non-Banking Financial Companies",
    ref_numbers=["RBI/TEST/1"],
    issued_on=date(2025, 1, 1),
    source_url="https://example.org/1",
)


def _captured(valid_from: date, text_sha: str = "t1") -> repo.CapturedVersion:
    return repo.CapturedVersion(
        valid_from=valid_from,
        title=DOC.title,
        raw_sha256="r" + text_sha,
        raw_path=f"xx/{text_sha}.html",
        text_sha256=text_sha,
        block_count=10,
        annex_links=[],
    )


async def test_upsert_document_is_idempotent(session: AsyncSession) -> None:
    id1, created1 = await repo.upsert_document(session, DOC)
    id2, created2 = await repo.upsert_document(session, DOC)
    assert id1 == id2
    assert (created1, created2) == (True, False)


async def test_version_outcomes(session: AsyncSession) -> None:
    doc_id, _ = await repo.upsert_document(session, DOC)

    assert await repo.record_captured_version(session, doc_id, _captured(date(2026, 1, 1))) == "new"
    await session.flush()
    assert (
        await repo.record_captured_version(session, doc_id, _captured(date(2026, 1, 1)))
        == "unchanged"
    )
    assert (
        await repo.record_captured_version(session, doc_id, _captured(date(2026, 1, 1), "t2"))
        == "silent_edit"
    )


async def test_listed_versions_then_validity_windows(session: AsyncSession) -> None:
    doc_id, _ = await repo.upsert_document(session, DOC)
    await repo.record_captured_version(session, doc_id, _captured(date(2026, 6, 1)))
    await session.flush()

    added = await repo.record_listed_versions(
        session, doc_id, [("6", date(2025, 1, 1)), ("7", date(2025, 7, 1))], DOC.title
    )
    assert added == 2
    # listing the same history again adds nothing
    assert (
        await repo.record_listed_versions(session, doc_id, [("6", date(2025, 1, 1))], DOC.title)
        == 0
    )

    await repo.recompute_validity(session, doc_id)
    session.expire_all()

    v = await repo.version_in_force(session, doc_id, date(2025, 3, 15))
    assert v is not None and v.valid_from == date(2025, 1, 1) and v.valid_to == date(2025, 7, 1)
    assert v.content_status is ContentStatus.UNAVAILABLE

    current = await repo.version_in_force(session, doc_id, date(2026, 11, 18))
    assert current is not None and current.valid_to is None
    assert current.content_status is ContentStatus.CAPTURED

    assert await repo.version_in_force(session, doc_id, date(2024, 12, 31)) is None


async def test_capturing_a_listed_version_fills_it(session: AsyncSession) -> None:
    doc_id, _ = await repo.upsert_document(session, DOC)
    await repo.record_listed_versions(session, doc_id, [("6", date(2025, 1, 1))], DOC.title)
    await session.flush()
    assert (
        await repo.record_captured_version(session, doc_id, _captured(date(2025, 1, 1))) == "filled"
    )
