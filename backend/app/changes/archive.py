"""Backfill a change from an archived copy of a direction (Internet Archive).

The crawler only started recording full text recently, so for older amendments we know
the version *dates* but not the earlier *text*. The Wayback Machine keeps public
snapshots of RBI pages; comparing one from before an amendment with the current text
gives a real paragraph diff of that amendment.

The event is recorded with kind "archive_backfill" and the snapshot's URL, so it is
never mistaken for a change the crawler detected live. One polite request per backfill;
this does not touch rbi.org.in at all.
"""

import logging
import uuid

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.changes.service import process_change_events
from app.db.models import ChangeEvent, Document, DocumentVersion
from app.ingestion.encoding import decode_html
from app.ingestion.raw_store import RawStore
from app.ingestion.sources import rbi

log = logging.getLogger(__name__)

WAYBACK = "https://web.archive.org/web/{timestamp}id_/{url}"  # id_ = original bytes


def snapshot_url(timestamp: str, url: str) -> str:
    return WAYBACK.format(timestamp=timestamp, url=url)


async def backfill_from_archive(
    session_factory: async_sessionmaker[AsyncSession],
    store: RawStore,
    source_key: str,
    timestamp: str,
    *,
    user_agent: str,
    transport: httpx.AsyncBaseTransport | None = None,
) -> uuid.UUID:
    """Diff the archived snapshot `timestamp` (YYYYMMDDhhmmss) of a direction against
    its current text. Returns the change event id (already diffed)."""
    async with session_factory() as s:
        row = (
            await s.execute(
                select(Document, DocumentVersion)
                .join(DocumentVersion, DocumentVersion.document_id == Document.id)
                .where(Document.source_key == source_key, DocumentVersion.raw_path.is_not(None))
                .order_by(DocumentVersion.valid_from.desc())
                .limit(1)
            )
        ).first()
    if row is None:
        raise ValueError(f"no stored text for document {source_key}")
    doc, current = row

    url = snapshot_url(timestamp, doc.source_url)
    async with httpx.AsyncClient(
        transport=transport, timeout=60.0, headers={"User-Agent": user_agent}
    ) as client:
        r = await client.get(url, follow_redirects=True)
    if r.status_code != 200:
        raise ValueError(f"archive returned HTTP {r.status_code} for {url}")
    page = rbi.parse_detail(decode_html(r.content))  # fail early on a non-direction page
    if not page.blocks:
        raise ValueError(f"archived page has no regulation text: {url}")

    blob = store.put(r.content, ".html")
    async with session_factory() as s, s.begin():
        event = ChangeEvent(
            document_id=doc.id,
            version_id=current.id,
            kind="archive_backfill",
            old_raw_path=blob.relative_path,
            new_raw_path=current.raw_path,
            diff={"archived_from": url},
        )
        s.add(event)
        await s.flush()
        event_id = event.id
    await process_change_events(session_factory, store)
    async with session_factory() as s, s.begin():  # keep the provenance after diffing
        event = await s.get_one(ChangeEvent, event_id)
        event.diff = {**event.diff, "archived_from": url}
    log.info("archive backfill %s: %s vs current", source_key, timestamp)
    return event_id
