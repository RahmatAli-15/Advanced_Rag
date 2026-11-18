"""End-to-end crawl against a fake RBI site and a real Postgres (rolled back afterwards)."""

import itertools
from pathlib import Path

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import CrawlRun, CrawlStatus, DocumentVersion
from app.ingestion.http import PoliteClient
from app.ingestion.pipeline import CrawlStats, crawl_master_directions
from app.ingestion.raw_store import RawStore

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).parent / "fixtures"


def _fake_rbi(challenge: bool = False) -> httpx.MockTransport:
    tokens = itertools.count()
    listing = (FIXTURES / "rbi_listing.html").read_bytes()
    detail = (FIXTURES / "rbi_detail.html").read_bytes()
    versions = (FIXTURES / "rbi_versions.html").read_bytes()

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/robots.txt":
            return httpx.Response(418)
        if challenge:
            return httpx.Response(200, text="testing whether you are a human visitor")
        if path.endswith("BS_ViewMasterDirections.aspx"):
            return httpx.Response(200, content=listing)
        if path.endswith("BS_ViewMasDirections.aspx"):
            # A per-request token, like the real site's session values.
            token = f"<!-- token {next(tokens)} -->".encode()
            return httpx.Response(200, content=detail + token)
        if path.endswith("NotificationPreVersion.aspx"):
            return httpx.Response(200, content=versions)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


async def _noop_sleep(_: float) -> None:
    return None


async def _crawl(
    factory: async_sessionmaker[AsyncSession], store: RawStore, challenge: bool = False
) -> CrawlStats:
    async with PoliteClient(
        "TestBot/1.0", transport=_fake_rbi(challenge), sleep=_noop_sleep
    ) as client:
        return await crawl_master_directions(
            client, factory, store, match="Know Your Customer", with_history=True
        )


async def test_recrawl_is_idempotent_and_ignores_volatile_bytes(
    session_factory: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    store = RawStore(tmp_path)

    first = await _crawl(session_factory, store)
    second = await _crawl(session_factory, store)

    assert (first.new_documents, first.new_versions) == (1, 1)
    assert first.listed_history_versions == 3
    assert (second.new_documents, second.unchanged) == (0, 1)
    assert second.listed_history_versions == 0
    # Bytes differed between fetches, text did not: only one raw file is kept.
    assert len(list(tmp_path.rglob("*.html"))) == 1

    async with session_factory() as s:
        assert await s.scalar(select(func.count()).select_from(DocumentVersion)) == 4


async def test_bot_challenge_aborts_the_run(
    session_factory: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    stats = await _crawl(session_factory, RawStore(tmp_path), challenge=True)

    assert stats.aborted and stats.fetched == 0
    async with session_factory() as s:
        run = await s.scalar(select(CrawlRun).order_by(CrawlRun.started_at.desc()))
        assert run is not None and run.status is CrawlStatus.FAILED
        assert run.error is not None and "bot protection" in run.error
