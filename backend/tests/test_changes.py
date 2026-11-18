"""Change detection: paragraph diff, the transactional outbox written by the crawler,
the silent-edit re-index fix, and the worker step that turns events into diffs."""

import itertools
from pathlib import Path

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.changes.diff import diff_paragraphs, paragraphs
from app.changes.service import process_change_events
from app.chunking.chunker import ChunkDraft
from app.chunking.pipeline import chunk_versions
from app.db.models import ChangeEvent, Chunk
from app.ingestion.http import PoliteClient
from app.ingestion.pipeline import CrawlStats, crawl_master_directions
from app.ingestion.raw_store import RawStore

FIXTURES = Path(__file__).parent / "fixtures"
OLD = b"Treasury Bills of the Government of India"
NEW = b"Treasury Bills and dated securities of the Government of India"


def _draft(refs: tuple[str, ...], text: str, ordinal: int = 0) -> ChunkDraft:
    return ChunkDraft(ordinal, "clause", ("Chapter I",), refs, None, text, (), 10)  # type: ignore[arg-type]


def test_diff_reports_added_removed_and_changed_paragraphs() -> None:
    old = {"1": "Short title.", "2": "CRR is 3%.", "3": "Repealed text."}
    new = {"1": "Short  title.", "2": "CRR is 4%.", "4": "New reporting rule."}
    d = diff_paragraphs(old, new)
    assert d["counts"] == {"added": 1, "removed": 1, "changed": 1}  # whitespace ignored
    assert d["changed"] == [{"para": "2", "before": "CRR is 3%.", "after": "CRR is 4%."}]
    assert d["added"][0]["para"] == "4" and d["removed"][0]["para"] == "3"


def test_a_clause_split_across_chunks_is_compared_as_one_paragraph() -> None:
    drafts = [_draft(("42(1)",), "First half,", 0), _draft(("42(1)",), "second  half.", 1)]
    assert paragraphs(drafts) == {"42(1)": "First half, second half."}


# --- crawl → outbox → diff, against a fake RBI site and a real Postgres --------------


def _site(edited: bool) -> httpx.MockTransport:
    tokens = itertools.count()
    listing = (FIXTURES / "rbi_listing.html").read_bytes()
    detail = (FIXTURES / "rbi_detail.html").read_bytes()
    if edited:  # RBI rewords a clause without changing the "Updated as on" date
        assert OLD in detail
        detail = detail.replace(OLD, NEW)
    versions = (FIXTURES / "rbi_versions.html").read_bytes()

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/robots.txt":
            return httpx.Response(418)
        if path.endswith("BS_ViewMasterDirections.aspx"):
            return httpx.Response(200, content=listing)
        if path.endswith("BS_ViewMasDirections.aspx"):
            return httpx.Response(200, content=detail + f"<!-- {next(tokens)} -->".encode())
        if path.endswith("NotificationPreVersion.aspx"):
            return httpx.Response(200, content=versions)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


async def _no_sleep(_: float) -> None:
    return None


async def _crawl(
    sf: async_sessionmaker[AsyncSession], store: RawStore, edited: bool = False
) -> CrawlStats:
    async with PoliteClient("TestBot/1.0", transport=_site(edited), sleep=_no_sleep) as c:
        return await crawl_master_directions(
            c, sf, store, match="Know Your Customer", with_history=True
        )


async def _events(sf: async_sessionmaker[AsyncSession]) -> list[ChangeEvent]:
    async with sf() as s:
        return list(await s.scalars(select(ChangeEvent).order_by(ChangeEvent.detected_at)))


@pytest.mark.integration
async def test_crawl_writes_change_events_and_worker_diffs_them(
    session_factory: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    store = RawStore(tmp_path)

    await _crawl(session_factory, store)
    [first] = await _events(session_factory)
    assert first.kind == "new_document" and first.old_raw_path is None

    await _crawl(session_factory, store)  # nothing changed: no new event
    assert len(await _events(session_factory)) == 1

    await chunk_versions(session_factory, store)
    async with session_factory() as s:
        chunks_before = await s.scalar(select(func.count()).select_from(Chunk))
    assert chunks_before

    stats = await _crawl(session_factory, store, edited=True)
    assert stats.silent_edits == 1
    events = await _events(session_factory)
    edit = events[-1]
    assert edit.kind == "silent_edit" and edit.old_raw_path == first.new_raw_path
    assert edit.new_raw_path != first.new_raw_path

    # Regression: the edited version's old chunks are gone, so indexing rebuilds them.
    async with session_factory() as s:
        assert await s.scalar(select(func.count()).select_from(Chunk)) == 0
    await chunk_versions(session_factory, store)
    async with session_factory() as s:
        texts = list(await s.scalars(select(Chunk.text)))
    assert any(NEW.decode() in t for t in texts)

    assert await process_change_events(session_factory, store) == 2
    events = await _events(session_factory)
    assert {e.status for e in events} == {"ready"}
    diff = events[-1].diff
    assert diff["counts"] == {"added": 0, "removed": 0, "changed": 1}
    assert diff["changed"][0]["para"] == "2(i)"
    assert NEW.decode() in diff["changed"][0]["after"]
    assert events[0].diff["counts"]["added"] > 0  # a new document: everything is new

    assert await process_change_events(session_factory, store) == 0  # idempotent
