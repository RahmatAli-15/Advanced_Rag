"""GET /corpus — what the system covers, so a UI can state its limits honestly."""

from datetime import date

from fastapi import APIRouter, Request
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Chunk, ContentStatus, Document, DocumentVersion

router = APIRouter(tags=["corpus"])


class CorpusStats(BaseModel):
    regulators: list[str]
    documents: int
    versions_captured: int
    chunks: int
    earliest_version: date | None
    latest_version: date | None


@router.get("/corpus", response_model=CorpusStats)
async def corpus(request: Request) -> CorpusStats:
    sf: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    captured = DocumentVersion.content_status == ContentStatus.CAPTURED
    async with sf() as s:
        regulators = (await s.scalars(select(Document.regulator).distinct())).all()
        documents = await s.scalar(select(func.count()).select_from(Document)) or 0
        versions, earliest, latest = (
            await s.execute(
                select(
                    func.count(),
                    func.min(DocumentVersion.valid_from),
                    func.max(DocumentVersion.valid_from),
                ).where(captured)
            )
        ).one()
        chunks = await s.scalar(select(func.count()).select_from(Chunk)) or 0
    return CorpusStats(
        regulators=sorted(regulators),
        documents=documents,
        versions_captured=versions,
        chunks=chunks,
        earliest_version=earliest,
        latest_version=latest,
    )
