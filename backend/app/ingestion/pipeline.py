"""Crawl pipeline for RBI Master Directions.

listing -> filter -> for each direction:
    fetch detail page -> store raw bytes -> parse -> upsert document + current version
    (optional) fetch version list -> record past versions -> recompute validity windows

Each document commits in its own transaction, so one bad page never loses the rest.
A bot-protection challenge aborts the whole run: we back off, we do not push through.
"""

import asyncio
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import CrawlRun, CrawlStatus
from app.ingestion import repository as repo
from app.ingestion.encoding import decode_html
from app.ingestion.http import BlockedByBotProtection, CrawlError, PoliteClient
from app.ingestion.raw_store import RawStore
from app.ingestion.sources import rbi

log = logging.getLogger(__name__)

SOURCE = "rbi.master_directions"


@dataclass
class CrawlStats:
    listed: int = 0
    matched: int = 0
    fetched: int = 0
    new_documents: int = 0
    new_versions: int = 0
    filled_versions: int = 0
    unchanged: int = 0
    silent_edits: int = 0
    listed_history_versions: int = 0
    errors: int = 0
    aborted: bool = False


def _matches(item: rbi.ListingItem, match: str | None) -> bool:
    if not match:
        return True
    needle = match.lower()
    return needle in item.title.lower() or needle in (item.category or "").lower()


async def crawl_master_directions(
    client: PoliteClient,
    session_factory: async_sessionmaker[AsyncSession],
    store: RawStore,
    *,
    match: str | None = None,
    limit: int | None = None,
    with_history: bool = False,
) -> CrawlStats:
    stats = CrawlStats()
    params = {"match": match, "limit": limit, "with_history": with_history}

    # Generate the id here rather than reading it back after commit: post-commit
    # attribute access would trigger an implicit (and, under asyncio, illegal) reload.
    run_id = uuid.uuid4()
    async with session_factory() as session:
        session.add(CrawlRun(id=run_id, source=SOURCE, status=CrawlStatus.RUNNING, params=params))
        await session.commit()

    error: str | None = None
    try:
        listing = await client.get(rbi.LISTING_URL)
        items = rbi.parse_listing(decode_html(listing.body))
        stats.listed = len(items)
        selected = [i for i in items if _matches(i, match)]
        stats.matched = len(selected)
        if limit is not None:
            selected = selected[:limit]

        for item in selected:
            try:
                await _ingest_one(client, session_factory, store, item, with_history, stats)
            except BlockedByBotProtection:
                raise
            except (CrawlError, rbi.PageLayoutChanged) as e:
                stats.errors += 1
                log.error("failed %s (%s): %s", item.source_id, item.title[:60], e)
    except BlockedByBotProtection as e:
        stats.aborted = True
        error = f"aborted: bot protection challenge at {e}"
        log.error(error)
    except Exception as e:
        stats.aborted = True
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        if stats.aborted:
            status = CrawlStatus.FAILED
        elif stats.errors:
            status = CrawlStatus.PARTIAL
        else:
            status = CrawlStatus.SUCCEEDED
        async with session_factory() as session:
            run = await session.get_one(CrawlRun, run_id)
            run.status = status
            run.stats = asdict(stats)
            run.error = error
            run.finished_at = datetime.now(UTC)
            await session.commit()
    return stats


async def _ingest_one(
    client: PoliteClient,
    session_factory: async_sessionmaker[AsyncSession],
    store: RawStore,
    item: rbi.ListingItem,
    with_history: bool,
    stats: CrawlStats,
) -> None:
    result = await client.get(item.url)
    stats.fetched += 1
    page = rbi.parse_detail(decode_html(result.body))
    text_sha = page.fingerprint

    valid_from = page.updated_as_on or page.issued_on or item.issued_on
    if valid_from is None:
        raise rbi.PageLayoutChanged(f"no version date found for {item.source_id}")

    history: list[rbi.VersionEntry] = []
    if with_history:
        hist_page = await client.get(rbi.versions_url(item.source_id))
        history = rbi.parse_versions(decode_html(hist_page.body))

    async with session_factory() as session, session.begin():
        doc_id, created = await repo.upsert_document(
            session,
            repo.DocumentData(
                regulator="RBI",
                doc_type="master_direction",
                source_key=item.source_id,
                title=page.title,
                category=item.category,
                ref_numbers=page.ref_numbers,
                issued_on=page.issued_on or item.issued_on,
                source_url=item.url,
            ),
        )
        stats.new_documents += int(created)

        # Pages embed per-request tokens, so raw bytes differ on every fetch. Compare
        # the extracted text instead, and keep raw bytes only when the text changed.
        if await repo.captured_text_sha(session, doc_id, valid_from) == text_sha:
            outcome = repo.VersionOutcome.UNCHANGED
        else:
            # If the transaction below fails, this blob is orphaned but harmless:
            # it is content-addressed and gets reused on the next attempt.
            blob = await asyncio.to_thread(store.put, result.body, ".html")
            outcome = await repo.record_captured_version(
                session,
                doc_id,
                repo.CapturedVersion(
                    valid_from=valid_from,
                    title=page.title,
                    raw_sha256=blob.sha256,
                    raw_path=blob.relative_path,
                    text_sha256=text_sha,
                    block_count=len(page.blocks),
                    annex_links=page.annex_links,
                ),
            )
        if outcome == repo.VersionOutcome.NEW:
            stats.new_versions += 1
        elif outcome == repo.VersionOutcome.FILLED:
            stats.filled_versions += 1
        elif outcome == repo.VersionOutcome.SILENT_EDIT:
            stats.silent_edits += 1
            log.warning("silent edit detected in %s", item.url)
        else:
            stats.unchanged += 1

        await session.flush()
        older = [(h.hist_id, h.updated_as_on) for h in history if h.updated_as_on != valid_from]
        stats.listed_history_versions += await repo.record_listed_versions(
            session, doc_id, older, page.title
        )
        await repo.recompute_validity(session, doc_id)

    log.info("%s %s -> %s", outcome, item.source_id, page.title[:70])
