"""Embed chunks that have no vector yet for the configured model.

Embeds `context + text`, so a chunk's vector also knows its document, heading path
and parent clause. Writes per batch: an interrupted run resumes where it stopped.
"""

import asyncio
import logging
from dataclasses import dataclass

from sqlalchemy import exists, insert, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Chunk, ChunkEmbedding
from app.embeddings.base import Embedder

log = logging.getLogger(__name__)


@dataclass
class EmbedStats:
    embedded: int = 0
    batches: int = 0
    seconds: float = 0.0


def embedding_input(context: str, text: str) -> str:
    return f"{context}\n\n{text}"


async def embed_pending_chunks(
    session_factory: async_sessionmaker[AsyncSession],
    embedder: Embedder,
    *,
    batch_size: int = 64,
) -> EmbedStats:
    stats = EmbedStats()
    loop = asyncio.get_running_loop()
    started = loop.time()
    has_vector = exists().where(
        ChunkEmbedding.chunk_id == Chunk.id, ChunkEmbedding.model == embedder.name
    )
    while True:
        async with session_factory() as s:
            rows = (
                await s.execute(
                    select(Chunk.id, Chunk.context, Chunk.text)
                    .where(~has_vector)
                    .order_by(Chunk.id)
                    .limit(batch_size)
                )
            ).all()
        if not rows:
            break
        vectors = await asyncio.to_thread(
            embedder.embed_documents, [embedding_input(r.context, r.text) for r in rows]
        )
        async with session_factory() as s, s.begin():
            await s.execute(
                insert(ChunkEmbedding),
                [
                    {"chunk_id": r.id, "model": embedder.name, "embedding": v}
                    for r, v in zip(rows, vectors, strict=True)
                ],
            )
        stats.embedded += len(rows)
        stats.batches += 1
        if stats.batches % 10 == 0:
            log.info("embedded %d chunks", stats.embedded)
    stats.seconds = round(loop.time() - started, 1)
    return stats
