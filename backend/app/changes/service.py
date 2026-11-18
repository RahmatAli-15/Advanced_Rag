"""Turn pending change events (written by the crawler, outbox-style) into paragraph diffs."""

import asyncio
import logging
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.changes.diff import diff_raw
from app.db.models import ChangeEvent
from app.ingestion.raw_store import RawStore

log = logging.getLogger(__name__)


async def process_change_events(
    session_factory: async_sessionmaker[AsyncSession], store: RawStore, limit: int = 100
) -> int:
    """Diff every pending event. Idempotent: only 'pending' rows are touched, and each
    is marked ready or failed in its own transaction. Returns events processed."""
    async with session_factory() as s:
        pending = (
            await s.execute(
                select(ChangeEvent.id, ChangeEvent.old_raw_path, ChangeEvent.new_raw_path)
                .where(ChangeEvent.status == "pending")
                .order_by(ChangeEvent.detected_at)
                .limit(limit)
            )
        ).all()

    for event_id, old_path, new_path in pending:
        status, diff, error = "ready", {}, None
        try:
            if new_path is None:
                raise ValueError("event has no new text")
            new = await asyncio.to_thread(store.get, new_path)
            old = await asyncio.to_thread(store.get, old_path) if old_path else None
            diff = await asyncio.to_thread(diff_raw, old, new)
        except Exception as e:  # one bad page must not block the other events
            status, error = "failed", f"{type(e).__name__}: {e}"[:1000]
            log.error("change event %s failed: %s", event_id, error)
        async with session_factory() as s, s.begin():
            event = await s.get_one(ChangeEvent, event_id)
            event.status, event.diff, event.error = status, diff, error
            event.processed_at = datetime.now(UTC)
    return len(pending)
