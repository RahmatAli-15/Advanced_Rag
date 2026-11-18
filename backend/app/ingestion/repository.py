"""Persistence for documents and versions. All writes are idempotent upserts."""

import uuid
from dataclasses import dataclass
from datetime import date

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ContentStatus, Document, DocumentVersion


@dataclass(frozen=True)
class DocumentData:
    regulator: str
    doc_type: str
    source_key: str
    title: str
    category: str | None
    ref_numbers: list[str]
    issued_on: date | None
    source_url: str


@dataclass(frozen=True)
class CapturedVersion:
    valid_from: date
    title: str
    raw_sha256: str
    raw_path: str
    text_sha256: str
    block_count: int
    annex_links: list[str]


class VersionOutcome:
    NEW = "new"
    UNCHANGED = "unchanged"
    FILLED = "filled"  # we previously only knew the date; now we have the text
    SILENT_EDIT = "silent_edit"  # same version date, different content


async def upsert_document(session: AsyncSession, d: DocumentData) -> tuple[uuid.UUID, bool]:
    """Insert or refresh a document. Returns (id, created)."""
    stmt = (
        insert(Document)
        .values(
            regulator=d.regulator,
            doc_type=d.doc_type,
            source_key=d.source_key,
            title=d.title,
            category=d.category,
            ref_numbers=d.ref_numbers,
            issued_on=d.issued_on,
            source_url=d.source_url,
        )
        .on_conflict_do_update(
            index_elements=["regulator", "doc_type", "source_key"],
            set_={
                "title": d.title,
                "category": d.category,
                "ref_numbers": d.ref_numbers,
                "issued_on": d.issued_on,
                "source_url": d.source_url,
                "last_seen_at": func.now(),
            },
        )
        # xmax = 0 only for a freshly inserted row: a cheap "was this an insert?" check
        .returning(Document.id, text("(xmax = 0) AS created"))
    )
    row = (await session.execute(stmt)).one()
    return row[0], bool(row[1])


async def captured_text_sha(
    session: AsyncSession, document_id: uuid.UUID, valid_from: date
) -> str | None:
    """Text hash of the version starting on `valid_from`, if we hold its text."""
    return await session.scalar(
        select(DocumentVersion.text_sha256).where(
            DocumentVersion.document_id == document_id,
            DocumentVersion.valid_from == valid_from,
            DocumentVersion.content_status == ContentStatus.CAPTURED,
        )
    )


async def record_captured_version(
    session: AsyncSession, document_id: uuid.UUID, v: CapturedVersion
) -> str:
    existing = await session.scalar(
        select(DocumentVersion).where(
            DocumentVersion.document_id == document_id,
            DocumentVersion.valid_from == v.valid_from,
        )
    )
    if existing is None:
        session.add(
            DocumentVersion(
                document_id=document_id,
                valid_from=v.valid_from,
                source_version_key="current",
                title=v.title,
                content_status=ContentStatus.CAPTURED,
                raw_sha256=v.raw_sha256,
                raw_path=v.raw_path,
                text_sha256=v.text_sha256,
                block_count=v.block_count,
                annex_links=v.annex_links,
            )
        )
        return VersionOutcome.NEW

    if existing.content_status is ContentStatus.UNAVAILABLE:
        outcome = VersionOutcome.FILLED
    elif existing.text_sha256 == v.text_sha256:
        return VersionOutcome.UNCHANGED
    else:
        # The page changed without a new "Updated as on" date. Keep the latest text;
        # the raw bytes of the old one stay in the content-addressed store.
        outcome = VersionOutcome.SILENT_EDIT

    existing.content_status = ContentStatus.CAPTURED
    existing.title = v.title
    existing.raw_sha256 = v.raw_sha256
    existing.raw_path = v.raw_path
    existing.text_sha256 = v.text_sha256
    existing.block_count = v.block_count
    existing.annex_links = v.annex_links
    existing.recorded_at = func.now()
    return outcome


async def record_listed_versions(
    session: AsyncSession, document_id: uuid.UUID, versions: list[tuple[str, date]], title: str
) -> int:
    """Record versions the regulator lists but whose text we cannot fetch. Returns rows added."""
    if not versions:
        return 0
    stmt = (
        insert(DocumentVersion)
        .values(
            [
                {
                    "id": uuid.uuid4(),
                    "document_id": document_id,
                    "valid_from": valid_from,
                    "source_version_key": f"hist:{key}",
                    "title": title,
                    "content_status": ContentStatus.UNAVAILABLE.value,
                    "annex_links": [],
                }
                for key, valid_from in versions
            ]
        )
        .on_conflict_do_nothing(index_elements=["document_id", "valid_from"])
        .returning(DocumentVersion.id)
    )
    return len((await session.execute(stmt)).all())


async def recompute_validity(session: AsyncSession, document_id: uuid.UUID) -> None:
    """Each version is valid until the next one starts; the latest has valid_to = NULL."""
    await session.execute(
        text(
            """
            UPDATE document_versions v
            SET valid_to = nxt.next_from
            FROM (
                SELECT id, LEAD(valid_from) OVER (ORDER BY valid_from) AS next_from
                FROM document_versions
                WHERE document_id = :doc_id
            ) nxt
            WHERE v.id = nxt.id AND v.valid_to IS DISTINCT FROM nxt.next_from
            """
        ),
        {"doc_id": document_id},
    )


async def version_in_force(
    session: AsyncSession, document_id: uuid.UUID, on: date
) -> DocumentVersion | None:
    """The version of a document that was in force on a given date."""
    return await session.scalar(
        select(DocumentVersion).where(
            DocumentVersion.document_id == document_id,
            DocumentVersion.valid_from <= on,
            (DocumentVersion.valid_to.is_(None)) | (DocumentVersion.valid_to > on),
        )
    )
