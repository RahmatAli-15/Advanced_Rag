"""Database models for regulatory documents.

Two time dimensions (bitemporal), because compliance asks two different questions:
  valid_from / valid_to   business time: when this version of the rule was in force
  recorded_at             system time:   when *we* captured it
"What did the rule say on 1 March?" uses valid time; "what did our system know when
it gave that answer?" uses recorded time.
"""

import uuid
from datetime import date, datetime
from enum import StrEnum
from typing import Any, ClassVar

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Computed,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

NAMING = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING)
    type_annotation_map: ClassVar[dict[Any, Any]] = {
        dict[str, Any]: JSONB,
        list[str]: JSONB,
        list[dict[str, Any]]: JSONB,
    }


class ContentStatus(StrEnum):
    CAPTURED = "captured"  # we hold the full text of this version
    UNAVAILABLE = "unavailable"  # the regulator lists this version but does not serve its text


class CrawlStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"  # finished, but some documents failed
    FAILED = "failed"


class Document(Base):
    """A logical regulatory document, e.g. one Master Direction across all its versions."""

    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("regulator", "doc_type", "source_key"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    regulator: Mapped[str] = mapped_column(String(16))
    doc_type: Mapped[str] = mapped_column(String(32))
    source_key: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(Text)
    ref_numbers: Mapped[list[str]] = mapped_column(default=list)
    issued_on: Mapped[date | None] = mapped_column(Date)
    source_url: Mapped[str] = mapped_column(Text)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    versions: Mapped[list["DocumentVersion"]] = relationship(
        back_populates="document", order_by="DocumentVersion.valid_from"
    )


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("document_id", "valid_from"),
        Index("ix_document_versions_validity", "document_id", "valid_from", "valid_to"),
        CheckConstraint("valid_to IS NULL OR valid_to > valid_from", name="valid_range"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Indexed via the leading column of ix_document_versions_validity.
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    valid_from: Mapped[date] = mapped_column(Date)
    valid_to: Mapped[date | None] = mapped_column(Date)  # NULL = currently in force
    source_version_key: Mapped[str | None] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(Text)
    content_status: Mapped[ContentStatus] = mapped_column(
        Enum(ContentStatus, name="content_status", values_callable=lambda e: [m.value for m in e])
    )
    raw_sha256: Mapped[str | None] = mapped_column(String(64))
    raw_path: Mapped[str | None] = mapped_column(Text)
    text_sha256: Mapped[str | None] = mapped_column(String(64))
    block_count: Mapped[int | None] = mapped_column(Integer)
    annex_links: Mapped[list[str]] = mapped_column(default=list)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    document: Mapped[Document] = relationship(back_populates="versions")


class Chunk(Base):
    """A retrieval unit: one clause (or part of one) of a captured document version.

    `context` is the header prepended for embedding and keyword search (document title,
    headings, paragraph refs, parent lead-in); `text` is the exact regulatory text shown
    and cited. `search_tsv` is maintained by Postgres, so keyword search can never drift
    out of sync with the text.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("version_id", "ordinal"),
        Index("ix_chunks_search_tsv", "search_tsv", postgresql_using="gin"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE")
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(16))
    heading_path: Mapped[list[str]] = mapped_column(default=list)
    para_refs: Mapped[list[str]] = mapped_column(default=list)
    context: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    amendments: Mapped[list[dict[str, Any]]] = mapped_column(default=list)
    chunker_version: Mapped[str] = mapped_column(String(32))
    search_tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', context || ' ' || text)", persisted=True),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


EMBEDDING_DIM = 1024  # BAAI/bge-m3. A model with another size needs a new column/table.


class ChunkEmbedding(Base):
    """One vector per (chunk, model).

    Keyed by model so a new embedding model can be backfilled next to the old one and
    switched over by configuration, with no window where search has no vectors.
    """

    __tablename__ = "chunk_embeddings"
    __table_args__ = (
        Index(
            "ix_chunk_embeddings_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    model: Mapped[str] = mapped_column(String(128), primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AnswerLog(Base):
    """Audit trail: every answer can be reproduced and explained later.

    Records exactly which clauses were retrieved and cited, the model and prompt
    version used, and the as-of date, so "why did the system say X on day Y?" has an
    answer. Chunk ids are stored as plain strings (no foreign key) so the log survives
    re-chunking.
    """

    __tablename__ = "answer_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question: Mapped[str] = mapped_column(Text)
    as_of: Mapped[date] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(32))
    answer: Mapped[str] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(128))
    prompt_version: Mapped[str] = mapped_column(String(32))
    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(default=list)
    cited_chunk_ids: Mapped[list[str]] = mapped_column(default=list)
    top_rerank_score: Mapped[float | None] = mapped_column(Float)
    usage: Mapped[dict[str, Any]] = mapped_column(default=dict)
    timings_ms: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


class CrawlRun(Base):
    """One execution of a crawler, for observability and auditing."""

    __tablename__ = "crawl_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source: Mapped[str] = mapped_column(String(64))
    status: Mapped[CrawlStatus] = mapped_column(
        Enum(CrawlStatus, name="crawl_status", values_callable=lambda e: [m.value for m in e])
    )
    params: Mapped[dict[str, Any]] = mapped_column(default=dict)
    stats: Mapped[dict[str, Any]] = mapped_column(default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
