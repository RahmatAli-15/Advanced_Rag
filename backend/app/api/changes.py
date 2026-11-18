"""GET /changes — recent changes detected in the regulations (newest first)."""

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import ChangeEvent, Document

router = APIRouter(tags=["changes"])


class ChangeOut(BaseModel):
    id: uuid.UUID
    document_title: str
    source_url: str
    kind: str  # new_document | new_version | silent_edit
    status: str  # pending | ready | failed
    detected_at: datetime
    counts: dict[str, int]
    changed: list[dict[str, Any]]
    added: list[dict[str, Any]]
    removed: list[dict[str, Any]]


@router.get("/changes", response_model=list[ChangeOut])
async def changes(request: Request, limit: int = Query(20, ge=1, le=100)) -> list[ChangeOut]:
    sf: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    async with sf() as s:
        rows = (
            await s.execute(
                select(ChangeEvent, Document.title, Document.source_url)
                .join(Document, Document.id == ChangeEvent.document_id)
                .order_by(ChangeEvent.detected_at.desc())
                .limit(limit)
            )
        ).all()
    return [
        ChangeOut(
            id=e.id,
            document_title=title,
            source_url=url,
            kind=e.kind,
            status=e.status,
            detected_at=e.detected_at,
            counts=e.diff.get("counts", {}),
            changed=e.diff.get("changed", []),
            added=e.diff.get("added", []),
            removed=e.diff.get("removed", []),
        )
        for e, title, url in rows
    ]
