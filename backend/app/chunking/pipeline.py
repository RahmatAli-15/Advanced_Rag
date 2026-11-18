"""Turn captured document versions into chunks.

Raw HTML is the source of truth: chunks are re-derived from stored bytes, so a better
parser or chunker is rolled out by bumping CHUNKER_VERSION and re-running, with no
re-crawl. Each version is re-chunked in its own transaction (delete + insert), so
readers never see a half-chunked version.
"""

import asyncio
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import delete, exists, insert, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.chunking.chunker import CHUNKER_VERSION, chunk_blocks, resolve_amendments
from app.db.models import Chunk, ContentStatus, Document, DocumentVersion
from app.ingestion.encoding import decode_html
from app.ingestion.raw_store import RawStore
from app.ingestion.sources import rbi

log = logging.getLogger(__name__)

# (regulator, doc_type) -> parser from raw HTML to the neutral document model
PARSERS: dict[tuple[str, str], Callable[[str], rbi.DetailPage]] = {
    ("RBI", "master_direction"): rbi.parse_detail,
}


@dataclass
class ChunkStats:
    versions: int = 0
    chunks: int = 0
    failed: int = 0


async def _pending_versions(
    session: AsyncSession, rebuild: bool
) -> list[tuple[uuid.UUID, str, str, str, str]]:
    current = exists().where(
        Chunk.version_id == DocumentVersion.id, Chunk.chunker_version == CHUNKER_VERSION
    )
    stmt = (
        select(
            DocumentVersion.id,
            DocumentVersion.raw_path,
            DocumentVersion.title,
            Document.regulator,
            Document.doc_type,
        )
        .join(Document, Document.id == DocumentVersion.document_id)
        .where(DocumentVersion.content_status == ContentStatus.CAPTURED)
    )
    if not rebuild:
        stmt = stmt.where(~current)
    rows = (await session.execute(stmt)).all()
    return [(r[0], r[1], r[2], r[3], r[4]) for r in rows if r[1] is not None]


async def chunk_versions(
    session_factory: async_sessionmaker[AsyncSession],
    store: RawStore,
    *,
    rebuild: bool = False,
) -> ChunkStats:
    stats = ChunkStats()
    async with session_factory() as session:
        pending = await _pending_versions(session, rebuild)

    for version_id, raw_path, title, regulator, doc_type in pending:
        parser = PARSERS.get((regulator, doc_type))
        if parser is None:
            log.warning("no parser for %s/%s", regulator, doc_type)
            continue
        try:
            raw = await asyncio.to_thread(store.get, raw_path)
            page = parser(decode_html(raw))
            drafts = chunk_blocks(page.blocks)
        except (OSError, rbi.PageLayoutChanged) as e:
            stats.failed += 1
            log.error("cannot chunk version %s: %s", version_id, e)
            continue

        rows = [
            {
                "id": uuid.uuid4(),
                "version_id": version_id,
                "ordinal": d.ordinal,
                "kind": d.kind,
                "heading_path": list(d.heading_path),
                "para_refs": list(d.para_refs),
                "context": d.context_header(title),
                "text": d.text,
                "token_count": d.token_count,
                "amendments": resolve_amendments(d, page.footnotes),
                "chunker_version": CHUNKER_VERSION,
            }
            for d in drafts
        ]
        async with session_factory() as session, session.begin():
            await session.execute(delete(Chunk).where(Chunk.version_id == version_id))
            if rows:
                await session.execute(insert(Chunk), rows)
        stats.versions += 1
        stats.chunks += len(rows)
        log.info("chunked %s: %d chunks", title[:70], len(rows))
    return stats
